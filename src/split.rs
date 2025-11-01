use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::Path;

use onnx_extractor::OnnxOperation;

use crate::error::SplitError;
use crate::fingerprint::Fingerprint;
use crate::model::{ModelGraph, ModelId};
use crate::shared::SharedTensorRegistry;
use crate::tensor::{TensorDescriptor, TensorKind, TensorMetadata};

pub struct SplitResult {
    pub backbone: GraphPartition,
    pub model_a_head: GraphPartition,
    pub model_b_head: GraphPartition,
    pub model_a_outputs: Vec<ModelOutputAssignment>,
    pub model_b_outputs: Vec<ModelOutputAssignment>,
    pub embeddings: Vec<EmbeddingDescriptor>,
}

#[derive(Debug, Clone)]
pub struct GraphPartition {
    pub operations: Vec<OperationSummary>,
    pub node_names: Vec<String>,
}

impl GraphPartition {
    fn new(operations: Vec<OperationSummary>) -> Self {
        let node_names = operations.iter().map(|op| op.name.clone()).collect();
        GraphPartition {
            operations,
            node_names,
        }
    }

    pub fn len(&self) -> usize {
        self.operations.len()
    }
}

#[derive(Debug, Clone)]
pub struct OperationSummary {
    pub index: usize,
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

impl OperationSummary {
    fn from_op(index: usize, op: &OnnxOperation) -> Self {
        OperationSummary {
            index,
            name: op.name.clone(),
            op_type: op.op_type.clone(),
            inputs: op.inputs.clone(),
            outputs: op.outputs.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelOutputAssignment {
    pub name: String,
    pub produced_by_backbone: bool,
}

#[derive(Debug, Clone)]
pub struct EmbeddingDescriptor {
    pub fingerprint_hex: String,
    pub tensors: HashMap<ModelId, TensorMetadata>,
    pub consumer_nodes: HashMap<ModelId, Vec<String>>,
}

pub fn split_models_from_files<P: AsRef<Path>>(
    model_a_path: P,
    model_b_path: P,
) -> Result<SplitResult, SplitError> {
    let bytes_a = fs::read(model_a_path)?;
    let bytes_b = fs::read(model_b_path)?;
    split_models_from_bytes(bytes_a, bytes_b)
}

pub fn split_models_from_bytes(
    model_a_bytes: Vec<u8>,
    model_b_bytes: Vec<u8>,
) -> Result<SplitResult, SplitError> {
    let mut model_a = ModelGraph::new(ModelId::A, model_a_bytes)?;
    let mut model_b = ModelGraph::new(ModelId::B, model_b_bytes)?;
    let mut registry = SharedTensorRegistry::new();

    let input_pairs = pair_descriptors(
        model_a.descriptors_for_inputs()?,
        model_b.descriptors_for_inputs()?,
    );
    for (desc_a, desc_b) in input_pairs {
        registry.add_pair(desc_a.fingerprint, desc_a, desc_b)?;
    }

    let initializer_pairs = pair_descriptors(
        model_a.descriptors_for_initializers()?,
        model_b.descriptors_for_initializers()?,
    );
    for (desc_a, desc_b) in initializer_pairs {
        registry.add_pair(desc_a.fingerprint, desc_a, desc_b)?;
    }

    loop {
        let ready_a = model_a.collect_ready_nodes(|id, name| registry.contains(id, name))?;
        let ready_b = model_b.collect_ready_nodes(|id, name| registry.contains(id, name))?;

        let mut node_pairs = Vec::new();
        for (fingerprint, indices_a) in &ready_a {
            if let Some(indices_b) = ready_b.get(fingerprint) {
                let count = indices_a.len().min(indices_b.len());
                for i in 0..count {
                    node_pairs.push((indices_a[i], indices_b[i]));
                }
            }
        }

        if node_pairs.is_empty() {
            break;
        }

        for (idx_a, idx_b) in node_pairs {
            register_node_pair(&mut registry, &mut model_a, idx_a, &mut model_b, idx_b)?;
        }
    }

    let backbone = build_partition(&model_a, |idx| model_a.is_node_shared(idx));
    let model_a_head = build_partition(&model_a, |idx| !model_a.is_node_shared(idx));
    let model_b_head = build_partition(&model_b, |idx| !model_b.is_node_shared(idx));

    let model_a_outputs = classify_outputs(&model_a);
    let model_b_outputs = classify_outputs(&model_b);

    let embeddings = extract_embeddings(&registry, &model_a, &model_b);

    Ok(SplitResult {
        backbone,
        model_a_head,
        model_b_head,
        model_a_outputs,
        model_b_outputs,
        embeddings,
    })
}

fn register_node_pair(
    registry: &mut SharedTensorRegistry,
    model_a: &mut ModelGraph,
    node_a: usize,
    model_b: &mut ModelGraph,
    node_b: usize,
) -> Result<(), SplitError> {
    model_a.mark_node_shared(node_a);
    model_b.mark_node_shared(node_b);

    let outputs_a = model_a.node_outputs(node_a).to_owned();
    let outputs_b = model_b.node_outputs(node_b).to_owned();

    if outputs_a.len() != outputs_b.len() {
        return Err(SplitError::structure(format!(
            "mismatched output count for nodes `{}` and `{}`",
            model_a.node_name(node_a),
            model_b.node_name(node_b)
        )));
    }

    for (out_a, out_b) in outputs_a.iter().zip(outputs_b.iter()) {
        if out_a.is_empty() && out_b.is_empty() {
            continue;
        }
        if out_a.is_empty() || out_b.is_empty() {
            return Err(SplitError::structure(format!(
                "optional output mismatch between `{}` and `{}`",
                model_a.node_name(node_a),
                model_b.node_name(node_b)
            )));
        }
        let desc_a = model_a.descriptor_for_tensor(out_a)?;
        let desc_b = model_b.descriptor_for_tensor(out_b)?;
        if desc_a.fingerprint != desc_b.fingerprint {
            return Err(SplitError::structure(format!(
                "output fingerprint mismatch for tensors `{}` and `{}`",
                out_a, out_b
            )));
        }
        registry.add_pair(desc_a.fingerprint, desc_a, desc_b)?;
    }

    Ok(())
}

fn build_partition<F>(model: &ModelGraph, predicate: F) -> GraphPartition
where
    F: Fn(usize) -> bool,
{
    let operations = model
        .model
        .operations
        .iter()
        .enumerate()
        .filter(|(idx, _)| predicate(*idx))
        .map(|(idx, op)| OperationSummary::from_op(idx, op))
        .collect::<Vec<_>>();
    GraphPartition::new(operations)
}

fn classify_outputs(model: &ModelGraph) -> Vec<ModelOutputAssignment> {
    let mut outputs = Vec::new();
    for name in model.graph_outputs() {
        if name.is_empty() {
            continue;
        }
        let produced_by_backbone = match model.tensor_kind(name) {
            Some(TensorKind::NodeOutput { node_index, .. }) => model.is_node_shared(*node_index),
            Some(TensorKind::Input) | Some(TensorKind::Initializer) => true,
            None => false,
        };
        outputs.push(ModelOutputAssignment {
            name: name.clone(),
            produced_by_backbone,
        });
    }
    outputs
}

fn extract_embeddings(
    registry: &SharedTensorRegistry,
    model_a: &ModelGraph,
    model_b: &ModelGraph,
) -> Vec<EmbeddingDescriptor> {
    let mut embeddings = Vec::new();

    for entry in registry.entries() {
        if let Some(descriptor_a) = entry.descriptor(ModelId::A) {
            if let Some(descriptor_b) = entry.descriptor(ModelId::B) {
                let mut tensors = HashMap::new();
                tensors.insert(ModelId::A, descriptor_a.metadata.clone());
                tensors.insert(ModelId::B, descriptor_b.metadata.clone());

                let mut consumer_nodes: HashMap<ModelId, Vec<String>> = HashMap::new();
                let mut has_head_consumer = false;

                if let TensorKind::NodeOutput { .. } = descriptor_a.kind {
                    let consumers = model_a
                        .consumers_of(descriptor_a.name())
                        .iter()
                        .filter(|&&idx| !model_a.is_node_shared(idx))
                        .map(|&idx| model_a.node_name(idx).to_string())
                        .collect::<Vec<_>>();
                    if !consumers.is_empty() {
                        has_head_consumer = true;
                        consumer_nodes.insert(ModelId::A, consumers);
                    }
                }

                if let TensorKind::NodeOutput { .. } = descriptor_b.kind {
                    let consumers = model_b
                        .consumers_of(descriptor_b.name())
                        .iter()
                        .filter(|&&idx| !model_b.is_node_shared(idx))
                        .map(|&idx| model_b.node_name(idx).to_string())
                        .collect::<Vec<_>>();
                    if !consumers.is_empty() {
                        has_head_consumer = true;
                        consumer_nodes.insert(ModelId::B, consumers);
                    }
                }

                if has_head_consumer {
                    embeddings.push(EmbeddingDescriptor {
                        fingerprint_hex: entry.fingerprint.to_hex(),
                        tensors,
                        consumer_nodes,
                    });
                }
            }
        }
    }

    embeddings
}

fn pair_descriptors(
    descriptors_a: Vec<TensorDescriptor>,
    descriptors_b: Vec<TensorDescriptor>,
) -> Vec<(TensorDescriptor, TensorDescriptor)> {
    let mut buckets_a: HashMap<Fingerprint, VecDeque<TensorDescriptor>> = HashMap::new();
    let mut buckets_b: HashMap<Fingerprint, VecDeque<TensorDescriptor>> = HashMap::new();

    for desc in descriptors_a {
        buckets_a
            .entry(desc.fingerprint)
            .or_default()
            .push_back(desc);
    }
    for desc in descriptors_b {
        buckets_b
            .entry(desc.fingerprint)
            .or_default()
            .push_back(desc);
    }

    let mut pairs = Vec::new();
    for (fingerprint, queue_a) in buckets_a.iter_mut() {
        if let Some(queue_b) = buckets_b.get_mut(fingerprint) {
            while let (Some(desc_a), Some(desc_b)) = (queue_a.pop_front(), queue_b.pop_front()) {
                pairs.push((desc_a, desc_b));
            }
        }
    }

    pairs
}
