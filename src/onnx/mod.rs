use crate::types::{Artifact, CanonicalValue, Format, Tensor};
use prost::Message;
use std::collections::BTreeMap;
use std::io::{Read, Seek};
use thiserror::Error;

mod onnx_proto {
    include!(concat!(env!("OUT_DIR"), "/onnx-proto/onnx.rs"));
}

use onnx_proto::ModelProto;

#[derive(Error, Debug)]
pub enum OnnxParserError {
    #[error("failed to parse ONNX: {0}")]
    ParseError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub fn parse_onnx<R: Read + Seek>(reader: &mut R) -> Result<Artifact, OnnxParserError> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;

    let model =
        ModelProto::decode(&*bytes).map_err(|e| OnnxParserError::ParseError(e.to_string()))?;

    let mut metadata = BTreeMap::new();
    let mut tensors = BTreeMap::new();

    if let Some(ir_version) = model.ir_version {
        metadata.insert("ir_version".to_string(), CanonicalValue::Int(ir_version));
    }

    if let Some(name) = model.producer_name {
        metadata.insert("producer_name".to_string(), CanonicalValue::String(name));
    }

    if let Some(version) = model.producer_version {
        metadata.insert(
            "producer_version".to_string(),
            CanonicalValue::String(version),
        );
    }

    if let Some(domain) = model.domain {
        metadata.insert("domain".to_string(), CanonicalValue::String(domain));
    }

    if let Some(model_version) = model.model_version {
        metadata.insert(
            "model_version".to_string(),
            CanonicalValue::Int(model_version),
        );
    }

    if !model.opset_import.is_empty() {
        let versions: Vec<String> = model
            .opset_import
            .iter()
            .filter_map(|op| op.version.map(|v| v.to_string()))
            .collect();
        metadata.insert(
            "opset_imports".to_string(),
            CanonicalValue::String(format!("{:?}", versions)),
        );
    }

    if let Some(graph) = model.graph {
        if !graph.node.is_empty() {
            let mut op_types: Vec<String> = graph
                .node
                .iter()
                .filter_map(|n| n.op_type.clone())
                .collect();
            op_types.sort();
            metadata.insert(
                "node_types".to_string(),
                CanonicalValue::String(format!("{:?}", op_types)),
            );
            metadata.insert(
                "node_count".to_string(),
                CanonicalValue::Int(graph.node.len() as i64),
            );
        }

        for init in &graph.initializer {
            let name = init.name.clone().unwrap_or_default();
            let dims: Vec<u64> = init.dims.iter().map(|&x| x as u64).collect();
            let dtype = onnx_dtype_str(init.data_type());
            let byte_length: u64 =
                dims.iter().product::<u64>() * dtype_size(init.data_type()) as u64;

            tensors.insert(
                name.clone(),
                Tensor {
                    name,
                    dtype,
                    shape: dims,
                    byte_length,
                },
            );
        }

        if !graph.input.is_empty() {
            let input_names: Vec<String> =
                graph.input.iter().filter_map(|i| i.name.clone()).collect();
            metadata.insert(
                "input_names".to_string(),
                CanonicalValue::String(format!("{:?}", input_names)),
            );
        }

        if !graph.output.is_empty() {
            let output_names: Vec<String> =
                graph.output.iter().filter_map(|o| o.name.clone()).collect();
            metadata.insert(
                "output_names".to_string(),
                CanonicalValue::String(format!("{:?}", output_names)),
            );
        }
    }

    let ir_version = metadata
        .get("ir_version")
        .and_then(|v| match v {
            CanonicalValue::Int(i) => Some(*i),
            _ => None,
        })
        .unwrap_or(0);

    Ok(Artifact {
        format: Format::Onnx,
        gguf_version: Some(ir_version),
        metadata,
        tensors,
    })
}

fn onnx_dtype_str(dtype: i32) -> String {
    match dtype {
        1 => "float32".to_string(),
        2 => "uint8".to_string(),
        3 => "int8".to_string(),
        4 => "uint16".to_string(),
        5 => "int16".to_string(),
        6 => "int32".to_string(),
        7 => "int64".to_string(),
        8 => "string".to_string(),
        9 => "bool".to_string(),
        10 => "float16".to_string(),
        11 => "float64".to_string(),
        12 => "uint32".to_string(),
        13 => "uint64".to_string(),
        14 => "complex64".to_string(),
        15 => "complex128".to_string(),
        16 => "bfloat16".to_string(),
        _ => format!("unknown_{}", dtype),
    }
}

fn dtype_size(dtype: i32) -> usize {
    match dtype {
        1 => 4,
        2 => 1,
        3 => 1,
        4 => 2,
        5 => 2,
        6 => 4,
        7 => 8,
        8 => 1,
        9 => 1,
        10 => 2,
        11 => 8,
        12 => 4,
        13 => 8,
        14 => 8,
        15 => 16,
        16 => 2,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_dtype_str() {
        assert_eq!(onnx_dtype_str(1), "float32");
        assert_eq!(onnx_dtype_str(7), "int64");
        assert_eq!(onnx_dtype_str(10), "float16");
    }

    #[test]
    fn test_dtype_size() {
        assert_eq!(dtype_size(1), 4);
        assert_eq!(dtype_size(7), 8);
        assert_eq!(dtype_size(10), 2);
    }
}
