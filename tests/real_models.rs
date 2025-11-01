use std::collections::{HashMap, HashSet};
use std::fs;

use onnx_protobuf::{GraphProto, ModelProto, TensorProto, Message};

use sonnx::split_models_from_bytes;

const MODEL_PATH: &str = "tests/data/squeezenet1.0-12.onnx";

#[test]
fn splits_real_model_after_weight_mutation() {
    let base_bytes = fs::read(MODEL_PATH).expect("model file present");
    let base_model = decode_model(&base_bytes);
    let base_graph = base_model
        .graph
        .as_ref()
        .expect("model contains a graph");
    let node_count = base_graph.node.len();
    assert!(
        node_count >= 4,
        "expected at least a few nodes, got {node_count}"
    );

    // Mutate weights consumed by nodes after the chosen cut.
    let cut_index = std::cmp::max(1, node_count / 2);
    let mut mutated_model = base_model.clone();
    let graph = mutated_model.graph.mut_or_insert_default();
    mutate_head_weights(graph, cut_index);
    let mutated_bytes = mutated_model
        .write_to_bytes()
        .expect("serialize mutated model");

    let result = split_models_from_bytes(base_bytes, mutated_bytes).expect("split succeeds");

    assert_eq!(
        result.backbone.operations.len(),
        cut_index,
        "expected backbone to retain the unmodified prefix"
    );
    assert_eq!(
        result.model_a_head.operations.len(),
        node_count - cut_index
    );
    assert_eq!(
        result.model_b_head.operations.len(),
        node_count - cut_index
    );

    // Ensure every head operation in model A still has its counterpart in B with matching names.
    let head_names_a: HashSet<&str> = result
        .model_a_head
        .operations
        .iter()
        .map(|op| op.name.as_str())
        .collect();
    let head_names_b: HashSet<&str> = result
        .model_b_head
        .operations
        .iter()
        .map(|op| op.name.as_str())
        .collect();
    assert_eq!(head_names_a, head_names_b);

    // The API should surface at least one embedding tensor feeding into the heads.
    assert!(
        result
            .embeddings
            .iter()
            .any(|emb| emb.consumer_nodes.len() == 2)
    );
}

fn decode_model(bytes: &[u8]) -> ModelProto {
    let mut model = ModelProto::new();
    model
        .merge_from_bytes(bytes)
        .expect("decode ModelProto");
    model
}

fn mutate_head_weights(graph: &mut GraphProto, cut_index: usize) {
    let consumer_map = build_consumer_map(graph);
    for initializer in graph.initializer.iter_mut() {
        if !initializer.name.is_empty() {
            if let Some(consumers) = consumer_map.get(initializer.name.as_str()) {
                if consumers.iter().any(|&idx| idx >= cut_index) {
                    perturb_tensor(initializer);
                }
            }
        }
    }
}

fn build_consumer_map(graph: &GraphProto) -> HashMap<String, Vec<usize>> {
    let mut map: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, node) in graph.node.iter().enumerate() {
        for input in &node.input {
            if !input.is_empty() {
                map.entry(input.to_string()).or_default().push(idx);
            }
        }
    }
    map
}

fn perturb_tensor(tensor: &mut TensorProto) {
    if let Some(first) = tensor.float_data.first_mut() {
        *first += 0.25;
        return;
    }

    if let Some(first) = tensor.double_data.first_mut() {
        *first += 0.25;
        return;
    }

    if let Some(first) = tensor.int32_data.first_mut() {
        *first = first.wrapping_add(1);
        return;
    }

    if let Some(first) = tensor.int64_data.first_mut() {
        *first = first.wrapping_add(1);
        return;
    }

    if !tensor.raw_data.is_empty() && tensor.raw_data.len() >= 4 {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&tensor.raw_data[..4]);
        let mut value = f32::from_le_bytes(bytes);
        value += 0.25;
        tensor.raw_data[..4].copy_from_slice(&value.to_le_bytes());
    }
}
