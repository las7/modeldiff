use crate::types::{Artifact, CanonicalValue, Format, Tensor};
use std::collections::BTreeMap;
use std::io::{Read, Seek};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OnnxParserError {
    #[error("invalid ONNX protobuf")]
    InvalidProtobuf,
    #[error("failed to parse ONNX: {0}")]
    ParseError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("node count exceeds maximum ({max}): {count}")]
    NodeCountTooLarge { count: usize, max: usize },
    #[error("initializer count exceeds maximum ({max}): {count}")]
    InitializerCountTooLarge { count: usize, max: usize },
    #[error("input/output count exceeds maximum ({max}): {count}")]
    IoCountTooLarge { count: usize, max: usize },
}

const MAX_NODES: usize = 100_000;
const MAX_INITIALIZERS: usize = 100_000;
const MAX_IO_COUNT: usize = 10_000;

pub fn parse_onnx<R: Read + Seek>(reader: &mut R) -> Result<Artifact, OnnxParserError> {
    let data = read_all_bytes(reader)?;

    let mut parser = ProtobufParser::new(&data);
    let mut metadata = BTreeMap::new();
    let mut tensors = BTreeMap::new();

    let mut ir_version: i64 = 0;

    while let Some((field, _wire_type)) = parser.read_tag() {
        match field {
            1 => {
                ir_version = parser.read_varint() as i64;
                metadata.insert("ir_version".to_string(), CanonicalValue::Int(ir_version));
            }
            2 => {
                parser.skip_field();
            }
            3 => {
                let name = parser.read_string();
                metadata.insert("producer_name".to_string(), CanonicalValue::String(name));
            }
            4 => {
                let version = parser.read_string();
                metadata.insert(
                    "producer_version".to_string(),
                    CanonicalValue::String(version),
                );
            }
            5 => {
                let domain = parser.read_string();
                metadata.insert("domain".to_string(), CanonicalValue::String(domain));
            }
            6 => {
                let model_version = parser.read_varint() as i64;
                metadata.insert(
                    "model_version".to_string(),
                    CanonicalValue::Int(model_version),
                );
            }
            11 => {
                let _graph_len = parser.read_varint() as usize;
                parse_graph(&mut parser, &data, &mut metadata, &mut tensors)?;
            }
            _ => {
                parser.skip_field();
            }
        }
    }

    Ok(Artifact {
        format: Format::Onnx,
        gguf_version: Some(ir_version),
        metadata,
        tensors,
    })
}

fn parse_graph(
    parser: &mut ProtobufParser,
    data: &[u8],
    metadata: &mut BTreeMap<String, CanonicalValue>,
    tensors: &mut BTreeMap<String, Tensor>,
) -> Result<(), OnnxParserError> {
    let mut node_types: Vec<String> = Vec::new();
    let mut node_count = 0;
    let mut input_names: Vec<String> = Vec::new();
    let mut output_names: Vec<String> = Vec::new();

    while let Some((field, _wire_type)) = parser.read_tag() {
        match field {
            1 => {
                node_count += 1;
                if node_count > MAX_NODES {
                    return Err(OnnxParserError::NodeCountTooLarge {
                        count: node_count,
                        max: MAX_NODES,
                    });
                }
                let _node_len = parser.read_varint() as usize;

                let mut op_type = String::new();
                let mut done = false;
                while !done {
                    match parser.read_tag() {
                        Some((3, _)) => {
                            op_type = parser.read_string();
                        }
                        Some(_) => {
                            parser.skip_field();
                        }
                        None => {
                            done = true;
                        }
                    }
                    if parser.position() > 10000 {
                        done = true;
                    }
                }
                if !op_type.is_empty() {
                    node_types.push(op_type);
                }
            }
            2 => {
                let init_len = parser.read_varint() as usize;
                let init_start = parser.position();
                let mut init_count = 0;

                while parser.position() - init_start < init_len && parser.position() < data.len() {
                    if let Some((1, _)) = parser.read_tag() {
                        let name = parser.read_string();
                        init_count += 1;
                        if init_count > MAX_INITIALIZERS {
                            return Err(OnnxParserError::InitializerCountTooLarge {
                                count: init_count,
                                max: MAX_INITIALIZERS,
                            });
                        }

                        let mut dims: Vec<i64> = Vec::new();
                        let mut data_type: i32 = 1;

                        let mut tensor_done = false;
                        while !tensor_done && parser.position() < data.len() {
                            match parser.read_tag() {
                                Some((2, _)) => {
                                    dims = parser.read_repeated_int64();
                                }
                                Some((3, _)) => {
                                    data_type = parser.read_varint() as i32;
                                }
                                Some(_) => {
                                    parser.skip_field();
                                }
                                None => {
                                    tensor_done = true;
                                }
                            }
                            if parser.position() - init_start > init_len {
                                tensor_done = true;
                            }
                        }

                        let dtype = onnx_dtype_str(data_type);
                        let shape: Vec<u64> = dims.iter().map(|&d| d as u64).collect();
                        let byte_length: u64 = shape
                            .iter()
                            .product::<u64>()
                            .saturating_mul(dtype_size(data_type) as u64);

                        tensors.insert(
                            name.clone(),
                            Tensor {
                                name,
                                dtype,
                                shape,
                                byte_length,
                            },
                        );
                    } else {
                        parser.skip_field();
                        if parser.position() - init_start >= init_len {
                            break;
                        }
                    }
                }
            }
            3 => {
                let input_len = parser.read_varint() as usize;
                let input_start = parser.position();
                let mut input_count = 0;

                while parser.position() - input_start < input_len && parser.position() < data.len()
                {
                    input_count += 1;
                    if input_count > MAX_IO_COUNT {
                        return Err(OnnxParserError::IoCountTooLarge {
                            count: input_count,
                            max: MAX_IO_COUNT,
                        });
                    }
                    match parser.read_tag() {
                        Some((1, _)) => {
                            let name = parser.read_string();
                            input_names.push(name);
                        }
                        Some(_) => {
                            parser.skip_field();
                        }
                        None => break,
                    }
                }
            }
            4 => {
                let output_len = parser.read_varint() as usize;
                let output_start = parser.position();
                let mut output_count = 0;

                while parser.position() - output_start < output_len
                    && parser.position() < data.len()
                {
                    output_count += 1;
                    if output_count > MAX_IO_COUNT {
                        return Err(OnnxParserError::IoCountTooLarge {
                            count: output_count,
                            max: MAX_IO_COUNT,
                        });
                    }
                    match parser.read_tag() {
                        Some((1, _)) => {
                            let name = parser.read_string();
                            output_names.push(name);
                        }
                        Some(_) => {
                            parser.skip_field();
                        }
                        None => break,
                    }
                }
            }
            _ => {
                parser.skip_field();
            }
        }
    }

    node_types.sort();
    metadata.insert(
        "node_types".to_string(),
        CanonicalValue::String(format!("{:?}", node_types)),
    );
    metadata.insert(
        "node_count".to_string(),
        CanonicalValue::Int(node_count as i64),
    );
    metadata.insert(
        "input_names".to_string(),
        CanonicalValue::String(format!("{:?}", input_names)),
    );
    metadata.insert(
        "output_names".to_string(),
        CanonicalValue::String(format!("{:?}", output_names)),
    );

    Ok(())
}

struct ProtobufParser {
    data: Vec<u8>,
    position: usize,
}

impl ProtobufParser {
    fn new(data: &[u8]) -> Self {
        Self {
            data: data.to_vec(),
            position: 0,
        }
    }

    fn position(&self) -> usize {
        self.position
    }

    fn read_tag(&mut self) -> Option<(u32, u8)> {
        if self.position >= self.data.len() {
            return None;
        }
        let tag = self.read_varint();
        if tag == 0 {
            return None;
        }
        let field = (tag >> 3) as u32;
        let wire_type = (tag & 0x7) as u8;
        Some((field, wire_type))
    }

    fn read_varint(&mut self) -> u64 {
        let mut result = 0u64;
        let mut shift = 0;
        loop {
            if self.position >= self.data.len() {
                break;
            }
            let byte = self.data[self.position];
            self.position += 1;
            result |= ((byte & 0x7F) as u64) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        result
    }

    fn read_string(&mut self) -> String {
        let len = self.read_varint() as usize;
        if self.position + len > self.data.len() {
            let remaining = &self.data[self.position..];
            let end = remaining.len().min(len);
            self.position += end;
            return String::from_utf8_lossy(&remaining[..end]).to_string();
        }
        let result =
            String::from_utf8_lossy(&self.data[self.position..self.position + len]).to_string();
        self.position += len;
        result
    }

    fn read_repeated_int64(&mut self) -> Vec<i64> {
        let start_pos = self.position;
        let mut values = Vec::new();

        loop {
            if self.position >= self.data.len() {
                break;
            }
            let byte = self.data[self.position];
            let field = (byte >> 3) as u32;
            let wire_type = byte & 0x7;

            if field != 2 || wire_type != 2 {
                self.position = start_pos;
                break;
            }

            let val = self.read_varint() as i64;
            values.push(val);
        }

        values
    }

    fn skip_field(&mut self) {
        if self.position >= self.data.len() {
            return;
        }
        let byte = self.data[self.position];
        let wire_type = byte & 0x7;

        match wire_type {
            0 => {
                self.read_varint();
            }
            1 => {
                self.position = (self.position + 8).min(self.data.len());
            }
            2 => {
                let len = self.read_varint() as usize;
                self.position = (self.position + len).min(self.data.len());
            }
            5 => {
                self.position = (self.position + 4).min(self.data.len());
            }
            _ => {}
        }
    }
}

fn read_all_bytes<R: Read>(reader: &mut R) -> Result<Vec<u8>, OnnxParserError> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn onnx_dtype_str(dtype: i32) -> String {
    match dtype {
        1 => "float32".to_string(),
        2 => "uint8".to_string(),
        3 => "int8".to_string(),
        6 => "int32".to_string(),
        7 => "int64".to_string(),
        10 => "float16".to_string(),
        11 => "float64".to_string(),
        12 => "uint32".to_string(),
        13 => "uint64".to_string(),
        14 => "complex64".to_string(),
        16 => "bfloat16".to_string(),
        _ => format!("unknown_{}", dtype),
    }
}

fn dtype_size(dtype: i32) -> usize {
    match dtype {
        1 => 4,
        2 => 1,
        3 => 1,
        6 => 4,
        7 => 8,
        10 => 2,
        11 => 8,
        12 => 4,
        13 => 8,
        14 => 8,
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
