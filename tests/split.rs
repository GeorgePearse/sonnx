use onnx_protobuf::{
    tensor_proto,
    tensor_shape_proto,
    type_proto,
    GraphProto,
    ModelProto,
    NodeProto,
    OperatorSetIdProto,
    TensorProto,
    TensorShapeProto,
    TypeProto,
    ValueInfoProto,
};
use protobuf::{Enum, Message, MessageField};

use sonnx::split_models_from_bytes;

fn build_test_model(head_op_type: &str, head_const: f32, output_name: &str) -> Vec<u8> {
    let mut model = ModelProto::new();
    model.ir_version = 7;

    let mut opset = OperatorSetIdProto::new();
    opset.version = 13;
    model.opset_import.push(opset);

    let mut graph = GraphProto::new();
    graph.name = "test_graph".to_string();

    graph.input.push(make_value_info("input"));
    graph.output.push(make_value_info(output_name));
    graph.value_info.push(make_value_info("identity_out"));
    graph.value_info.push(make_value_info("backbone_out"));

    graph
        .initializer
        .push(make_scalar_tensor("head_weight", head_const));

    graph.node.push(make_node(
        "backbone_identity",
        "Identity",
        &["input"],
        &["identity_out"],
    ));
    graph.node.push(make_node(
        "backbone_relu",
        "Relu",
        &["identity_out"],
        &["backbone_out"],
    ));
    graph.node.push(make_node(
        "head",
        head_op_type,
        &["backbone_out", "head_weight"],
        &[output_name],
    ));

    model.graph = MessageField::some(graph);
    model.write_to_bytes().expect("serialize model")
}

fn make_node(name: &str, op_type: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
    let mut node = NodeProto::new();
    node.name = name.to_string();
    node.op_type = op_type.to_string();
    for input in inputs {
        node.input.push((*input).to_string());
    }
    for output in outputs {
        node.output.push((*output).to_string());
    }
    node
}

fn make_value_info(name: &str) -> ValueInfoProto {
    let mut value = ValueInfoProto::new();
    value.name = name.to_string();

    let mut tensor_type = type_proto::Tensor::new();
    tensor_type.elem_type = tensor_proto::DataType::FLOAT.value();

    let mut shape = TensorShapeProto::new();
    let mut dim = tensor_shape_proto::Dimension::new();
    dim.value = Some(tensor_shape_proto::dimension::Value::DimValue(1));
    shape.dim.push(dim);
    tensor_type.shape = MessageField::some(shape);

    let mut type_proto = TypeProto::new();
    type_proto.set_tensor_type(tensor_type);
    value.type_ = MessageField::some(type_proto);

    value
}

fn make_scalar_tensor(name: &str, value: f32) -> TensorProto {
    let mut tensor = TensorProto::new();
    tensor.name = name.to_string();
    tensor.data_type = tensor_proto::DataType::FLOAT.value();
    tensor.dims.push(1);
    tensor.float_data.push(value);
    tensor
}

#[test]
fn split_identifies_shared_backbone_and_embeddings() {
    let model_a = build_test_model("Add", 1.0, "output_a");
    let model_b = build_test_model("Mul", 2.0, "output_b");

    let result = split_models_from_bytes(model_a, model_b).expect("split models");

    assert_eq!(result.backbone.operations.len(), 2);
    let backbone_ops: Vec<&str> = result
        .backbone
        .operations
        .iter()
        .map(|op| op.op_type.as_str())
        .collect();
    assert_eq!(backbone_ops, vec!["Identity", "Relu"]);

    assert_eq!(result.model_a_head.operations.len(), 1);
    assert_eq!(result.model_a_head.operations[0].op_type, "Add");
    assert_eq!(result.model_b_head.operations.len(), 1);
    assert_eq!(result.model_b_head.operations[0].op_type, "Mul");

    let embedding_names: Vec<&str> = result
        .embeddings
        .iter()
        .flat_map(|emb| emb.tensors.values())
        .map(|meta| meta.name.as_str())
        .collect();
    assert!(embedding_names.contains(&"backbone_out"));

    let model_a_output = result
        .model_a_outputs
        .iter()
        .find(|out| out.name == "output_a")
        .expect("output_a present");
    assert!(!model_a_output.produced_by_backbone);

    let model_b_output = result
        .model_b_outputs
        .iter()
        .find(|out| out.name == "output_b")
        .expect("output_b present");
    assert!(!model_b_output.produced_by_backbone);
}
