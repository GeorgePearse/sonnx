use thiserror::Error;

#[derive(Debug, Error)]
pub enum SplitError {
    #[error("ONNX model is missing a graph section")]
    MissingGraph,

    #[error("Tensor `{0}` not found in model")]
    MissingTensor(String),

    #[error("Unsupported ONNX feature: {0}")]
    Unsupported(String),

    #[error("Model structures differ: {0}")]
    StructureMismatch(String),

    #[error("ONNX parsing error: {0}")]
    Onnx(#[from] onnx_extractor::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl SplitError {
    pub(crate) fn structure(msg: impl Into<String>) -> Self {
        SplitError::StructureMismatch(msg.into())
    }

    pub(crate) fn missing_tensor(name: impl Into<String>) -> Self {
        SplitError::MissingTensor(name.into())
    }
}
