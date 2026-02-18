use crate::types::{Artifact, CanonicalValue, Format, Tensor};
use std::collections::BTreeMap;
use std::io::{Read, Seek};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SafetensorsParserError {
    #[error("invalid safetensors JSON header")]
    InvalidHeader,
    #[error("header size exceeds maximum allowed ({max} bytes): {size} bytes")]
    HeaderTooLarge { size: usize, max: usize },
    #[error("invalid tensor length for '{name}': offset {offset} > {end}")]
    InvalidByteLength { name: String, offset: u64, end: u64 },
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

const MAX_HEADER_SIZE: usize = 100 * 1024 * 1024; // 100MB

pub fn parse_safetensors<R: Read + Seek>(
    reader: &mut R,
) -> Result<Artifact, SafetensorsParserError> {
    let header_size = read_header_size(reader)?;
    let mut header_bytes = vec![0u8; header_size];
    reader.read_exact(&mut header_bytes)?;

    let json: serde_json::Value = serde_json::from_slice(&header_bytes)?;

    let mut metadata = BTreeMap::new();
    let mut tensors = BTreeMap::new();

    if let Some(obj) = json.as_object() {
        for (key, value) in obj {
            if key == "__metadata__" {
                if let Some(meta_obj) = value.as_object() {
                    for (k, v) in meta_obj {
                        if let Some(v_str) = v.as_str() {
                            metadata.insert(k.clone(), CanonicalValue::String(v_str.to_string()));
                        }
                    }
                }
            } else if let Some(tensor_obj) = value.as_object() {
                let dtype = tensor_obj
                    .get("dtype")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let shape: Vec<u64> = tensor_obj
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect())
                    .unwrap_or_default();
                let data_offsets = tensor_obj
                    .get("data_offsets")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        let a = arr.first().and_then(|v| v.as_u64()).unwrap_or(0);
                        let b = arr.get(1).and_then(|v| v.as_u64()).unwrap_or(0);
                        [a, b]
                    })
                    .unwrap_or([0, 0]);
                let offset = data_offsets[0];
                let end = data_offsets[1];
                if end <= offset {
                    return Err(SafetensorsParserError::InvalidByteLength {
                        name: key.clone(),
                        offset,
                        end,
                    });
                }
                let byte_length = end - offset;

                tensors.insert(
                    key.clone(),
                    Tensor {
                        name: key.clone(),
                        dtype,
                        shape,
                        byte_length,
                    },
                );
            }
        }
    }

    Ok(Artifact {
        format: Format::Safetensors,
        gguf_version: None,
        metadata,
        tensors,
    })
}

fn read_header_size<R: Read + Seek>(reader: &mut R) -> Result<usize, SafetensorsParserError> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    let size = u64::from_le_bytes(buf) as usize;
    if size > MAX_HEADER_SIZE {
        return Err(SafetensorsParserError::HeaderTooLarge {
            size,
            max: MAX_HEADER_SIZE,
        });
    }
    Ok(size)
}
