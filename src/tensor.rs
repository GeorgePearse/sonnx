use crate::fingerprint::Fingerprint;
use onnx_extractor::DataType;

#[derive(Clone, Debug, PartialEq)]
pub struct TensorMetadata {
    pub name: String,
    pub data_type: Option<DataType>,
    pub shape: Vec<i64>,
}

impl TensorMetadata {
    pub fn new(name: impl Into<String>, data_type: Option<DataType>, shape: Vec<i64>) -> Self {
        TensorMetadata {
            name: name.into(),
            data_type,
            shape,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorKind {
    Input,
    Initializer,
    NodeOutput {
        node_index: usize,
        output_index: usize,
    },
}

#[derive(Clone, Debug)]
pub struct TensorDescriptor {
    pub fingerprint: Fingerprint,
    pub metadata: TensorMetadata,
    pub kind: TensorKind,
}

impl TensorDescriptor {
    pub fn name(&self) -> &str {
        &self.metadata.name
    }
}
