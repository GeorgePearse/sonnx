use std::collections::HashMap;

use crate::error::SplitError;
use crate::fingerprint::Fingerprint;
use crate::model::ModelId;
use crate::tensor::TensorDescriptor;

#[derive(Debug, Clone)]
pub struct SharedTensorEntry {
    pub fingerprint: Fingerprint,
    descriptors: HashMap<ModelId, TensorDescriptor>,
}

impl SharedTensorEntry {
    pub fn descriptor(&self, model: ModelId) -> Option<&TensorDescriptor> {
        self.descriptors.get(&model)
    }
}

pub struct SharedTensorRegistry {
    entries: Vec<SharedTensorEntry>,
    name_to_entry: HashMap<ModelId, HashMap<String, usize>>,
}

impl SharedTensorRegistry {
    pub fn new() -> Self {
        SharedTensorRegistry {
            entries: Vec::new(),
            name_to_entry: HashMap::new(),
        }
    }

    pub fn contains(&self, model: ModelId, tensor_name: &str) -> bool {
        self.name_to_entry
            .get(&model)
            .and_then(|map| map.get(tensor_name))
            .is_some()
    }

    pub fn add_pair(
        &mut self,
        fingerprint: Fingerprint,
        a: TensorDescriptor,
        b: TensorDescriptor,
    ) -> Result<usize, SplitError> {
        let existing_a = self
            .name_to_entry
            .get(&ModelId::A)
            .and_then(|map| map.get(a.name()))
            .copied();
        let existing_b = self
            .name_to_entry
            .get(&ModelId::B)
            .and_then(|map| map.get(b.name()))
            .copied();

        match (existing_a, existing_b) {
            (Some(idx_a), Some(idx_b)) if idx_a == idx_b => {
                return Ok(idx_a);
            }
            (Some(_), _) | (_, Some(_)) => {
                return Err(SplitError::structure(format!(
                    "tensor `{}` or `{}` already registered as shared",
                    a.name(),
                    b.name()
                )));
            }
            _ => {}
        }

        let index = self.entries.len();
        let mut descriptors = HashMap::new();
        descriptors.insert(ModelId::A, a);
        descriptors.insert(ModelId::B, b);

        self.entries.push(SharedTensorEntry {
            fingerprint,
            descriptors,
        });

        let name_a = self.entries[index]
            .descriptor(ModelId::A)
            .unwrap()
            .name()
            .to_string();
        let name_b = self.entries[index]
            .descriptor(ModelId::B)
            .unwrap()
            .name()
            .to_string();

        self.name_to_entry
            .entry(ModelId::A)
            .or_default()
            .insert(name_a, index);
        self.name_to_entry
            .entry(ModelId::B)
            .or_default()
            .insert(name_b, index);

        Ok(index)
    }

    pub fn entries(&self) -> &[SharedTensorEntry] {
        &self.entries
    }
}
