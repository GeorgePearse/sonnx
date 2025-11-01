use std::collections::{HashMap, HashSet};

use onnx_extractor::{AttributeValue, DataType, OnnxModel, OnnxTensor, TensorData};

use crate::error::SplitError;
use crate::fingerprint::{Fingerprint, FingerprintBuilder};
use crate::tensor::{TensorDescriptor, TensorKind, TensorMetadata};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum ModelId {
    A,
    B,
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelId::A => write!(f, "model_a"),
            ModelId::B => write!(f, "model_b"),
        }
    }
}

pub struct ModelGraph {
    id: ModelId,
    pub(crate) model: OnnxModel,
    node_fingerprints: Vec<Option<Fingerprint>>,
    tensor_fingerprints: HashMap<String, Fingerprint>,
    tensor_stack: HashSet<String>,
    tensor_kind: HashMap<String, TensorKind>,
    tensor_metadata_cache: HashMap<String, TensorMetadata>,
    initializer_names: HashSet<String>,
    consumers: HashMap<String, Vec<usize>>,
    shared_nodes: HashSet<usize>,
}

impl ModelGraph {
    pub fn new(id: ModelId, bytes: Vec<u8>) -> Result<Self, SplitError> {
        let model = OnnxModel::load_from_bytes(bytes)?;

        let mut tensor_kind = HashMap::new();
        for input in &model.inputs {
            if !input.is_empty() {
                tensor_kind.insert(input.clone(), TensorKind::Input);
            }
        }

        let mut initializer_names = HashSet::new();
        for tensor in model.get_weight_tensors() {
            let name = tensor.name().to_string();
            if !name.is_empty() {
                tensor_kind.insert(name.clone(), TensorKind::Initializer);
                initializer_names.insert(name);
            }
        }

        let mut consumers: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, op) in model.operations.iter().enumerate() {
            for (output_index, output) in op.outputs.iter().enumerate() {
                if !output.is_empty() {
                    tensor_kind.insert(
                        output.clone(),
                        TensorKind::NodeOutput {
                            node_index: idx,
                            output_index,
                        },
                    );
                }
            }
            for input in &op.inputs {
                if !input.is_empty() {
                    consumers.entry(input.clone()).or_default().push(idx);
                }
            }
        }

        let node_count = model.operations.len();

        Ok(ModelGraph {
            id,
            model,
            node_fingerprints: vec![None; node_count],
            tensor_fingerprints: HashMap::new(),
            tensor_stack: HashSet::new(),
            tensor_kind,
            tensor_metadata_cache: HashMap::new(),
            initializer_names,
            consumers,
            shared_nodes: HashSet::new(),
        })
    }

    pub fn mark_node_shared(&mut self, node_index: usize) {
        self.shared_nodes.insert(node_index);
    }

    pub fn is_node_shared(&self, node_index: usize) -> bool {
        self.shared_nodes.contains(&node_index)
    }

    pub fn node_name(&self, node_index: usize) -> &str {
        self.model.operations[node_index].name.as_str()
    }

    pub fn node_outputs(&self, node_index: usize) -> Vec<String> {
        self.model.operations[node_index].outputs.clone()
    }

    pub fn graph_outputs(&self) -> &[String] {
        &self.model.outputs
    }

    pub fn consumers_of(&self, tensor: &str) -> &[usize] {
        self.consumers
            .get(tensor)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn descriptor_for_tensor(&mut self, name: &str) -> Result<TensorDescriptor, SplitError> {
        let kind = self
            .tensor_kind
            .get(name)
            .cloned()
            .ok_or_else(|| SplitError::missing_tensor(name))?;
        let fingerprint = self.tensor_fingerprint(name)?;
        let metadata = self.tensor_metadata(name);
        Ok(TensorDescriptor {
            fingerprint,
            metadata,
            kind,
        })
    }

    pub fn descriptors_for_inputs(&mut self) -> Result<Vec<TensorDescriptor>, SplitError> {
        let mut descriptors = Vec::new();
        for name in self.model.inputs.clone() {
            if !name.is_empty() {
                descriptors.push(self.descriptor_for_tensor(&name)?);
            }
        }
        Ok(descriptors)
    }

    pub fn descriptors_for_initializers(&mut self) -> Result<Vec<TensorDescriptor>, SplitError> {
        let mut names: Vec<String> = self.initializer_names.iter().cloned().collect();
        names.sort();
        let mut descriptors = Vec::with_capacity(names.len());
        for name in names {
            descriptors.push(self.descriptor_for_tensor(&name)?);
        }
        Ok(descriptors)
    }

    pub fn tensor_kind(&self, name: &str) -> Option<&TensorKind> {
        self.tensor_kind.get(name)
    }

    pub fn collect_ready_nodes<F>(
        &mut self,
        mut is_shared: F,
    ) -> Result<HashMap<Fingerprint, Vec<usize>>, SplitError>
    where
        F: FnMut(ModelId, &str) -> bool,
    {
        let mut ready: HashMap<Fingerprint, Vec<usize>> = HashMap::new();

        let op_count = self.model.operations.len();
        for idx in 0..op_count {
            if self.shared_nodes.contains(&idx) {
                continue;
            }

            let inputs = self.model.operations[idx].inputs.clone();
            let all_inputs_shared = inputs
                .iter()
                .all(|input| input.is_empty() || is_shared(self.id, input));

            if !all_inputs_shared {
                continue;
            }

            let fingerprint = self.node_fingerprint(idx)?;
            ready.entry(fingerprint).or_default().push(idx);
        }

        Ok(ready)
    }

    pub fn tensor_metadata(&mut self, name: &str) -> TensorMetadata {
        if let Some(meta) = self.tensor_metadata_cache.get(name) {
            return meta.clone();
        }

        let metadata = if let Some(tensor) = self.model.tensors.get(name) {
            tensor_metadata_from_tensor(name, tensor)
        } else {
            TensorMetadata::new(name, None, Vec::new())
        };

        self.tensor_metadata_cache
            .insert(name.to_string(), metadata.clone());
        metadata
    }

    fn tensor_fingerprint(&mut self, name: &str) -> Result<Fingerprint, SplitError> {
        if let Some(fp) = self.tensor_fingerprints.get(name) {
            return Ok(*fp);
        }

        if !self.tensor_stack.insert(name.to_string()) {
            return Err(SplitError::structure(format!(
                "Cycle detected while fingerprinting tensor `{}`",
                name
            )));
        }

        let fingerprint = match self
            .tensor_kind
            .get(name)
            .cloned()
            .ok_or_else(|| SplitError::missing_tensor(name))?
        {
            TensorKind::Input => self.compute_input_fingerprint(name)?,
            TensorKind::Initializer => self.compute_initializer_fingerprint(name)?,
            TensorKind::NodeOutput {
                node_index,
                output_index,
            } => {
                let node_fp = self.node_fingerprint(node_index)?;
                node_fp.derive(b"node_output", output_index as u64)
            }
        };

        self.tensor_stack.remove(name);
        self.tensor_fingerprints
            .insert(name.to_string(), fingerprint);
        Ok(fingerprint)
    }

    fn compute_input_fingerprint(&mut self, name: &str) -> Result<Fingerprint, SplitError> {
        let metadata = self.tensor_metadata(name);
        let mut builder = FingerprintBuilder::new(b"input");
        if let Some(dt) = metadata.data_type {
            builder.update_u64(dt as u64);
        } else {
            builder.update_u64(0);
        }
        builder.update_u64(metadata.shape.len() as u64);
        for dim in &metadata.shape {
            builder.update_i64(*dim);
        }
        Ok(builder.finish())
    }

    fn compute_initializer_fingerprint(&mut self, name: &str) -> Result<Fingerprint, SplitError> {
        if !self.initializer_names.contains(name) {
            return Err(SplitError::missing_tensor(name));
        }
        let tensor = self
            .model
            .tensors
            .get(name)
            .ok_or_else(|| SplitError::missing_tensor(name))?;
        hash_tensor_data(tensor)
    }

    fn node_fingerprint(&mut self, node_index: usize) -> Result<Fingerprint, SplitError> {
        if let Some(fp) = self.node_fingerprints[node_index] {
            return Ok(fp);
        }

        let op = &self.model.operations[node_index];
        let op_type = op.op_type.clone();
        let inputs = op.inputs.clone();
        let outputs = op.outputs.clone();

        let mut builder = FingerprintBuilder::new(b"node");
        builder.update_str(op_type.as_str());

        let mut attr_hashes = Vec::new();
        for (name, value) in &op.attributes {
            let hash = hash_attribute_value(value)?;
            attr_hashes.push((name.clone(), hash));
        }
        attr_hashes.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, hash) in attr_hashes {
            builder.update_str(&name);
            builder.update_bytes(hash.as_bytes());
        }

        builder.update_u64(inputs.len() as u64);
        for input in &inputs {
            if input.is_empty() {
                builder.update_bytes(b"<empty>");
                continue;
            }
            let fp = self.tensor_fingerprint(input)?;
            builder.update_bytes(fp.as_bytes());
        }

        builder.update_u64(outputs.len() as u64);
        for output in &outputs {
            builder.update_str(output);
        }

        let fingerprint = builder.finish();
        self.node_fingerprints[node_index] = Some(fingerprint);
        Ok(fingerprint)
    }
}

fn tensor_metadata_from_tensor(name: &str, tensor: &OnnxTensor) -> TensorMetadata {
    let shape = tensor.shape().to_vec();
    let data_type = match tensor.data_type() {
        DataType::Undefined => None,
        other => Some(other),
    };
    TensorMetadata::new(name, data_type, shape)
}

fn hash_tensor_data(tensor: &OnnxTensor) -> Result<Fingerprint, SplitError> {
    let mut builder = FingerprintBuilder::new(b"tensor");
    builder.update_u64(tensor.data_type() as u64);
    builder.update_u64(tensor.shape().len() as u64);
    for dim in tensor.shape() {
        builder.update_i64(*dim);
    }

    match tensor.data()? {
        TensorData::Raw(bytes) => builder.update_bytes(bytes.as_ref()),
        TensorData::Numeric(cow) => builder.update_bytes(cow.as_ref()),
        TensorData::Strings(parts) => {
            builder.update_u64(parts.len() as u64);
            for part in parts {
                builder.update_bytes(part.as_ref());
            }
        }
    }

    Ok(builder.finish())
}

fn hash_attribute_value(value: &AttributeValue) -> Result<Fingerprint, SplitError> {
    let mut builder = FingerprintBuilder::new(b"attr_value");
    match value {
        AttributeValue::Int(v) => builder.update_i64(*v),
        AttributeValue::Float(v) => builder.update_f32(*v),
        AttributeValue::String(v) => builder.update_str(v),
        AttributeValue::Tensor(t) => {
            let hash = hash_tensor_data(t.as_ref())?;
            builder.update_bytes(hash.as_bytes());
        }
        AttributeValue::Ints(values) => {
            builder.update_u64(values.len() as u64);
            for v in values {
                builder.update_i64(*v);
            }
        }
        AttributeValue::Floats(values) => {
            builder.update_u64(values.len() as u64);
            for v in values {
                builder.update_f32(*v);
            }
        }
        AttributeValue::Strings(values) => {
            builder.update_u64(values.len() as u64);
            for v in values {
                builder.update_str(v);
            }
        }
    }
    Ok(builder.finish())
}
