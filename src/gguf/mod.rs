use crate::types::{Artifact, CanonicalValue, Format, Tensor};
use std::collections::BTreeMap;
use std::io::{Read, Seek};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GGUFParserError {
    #[error("unable to parse GGUF header")]
    InvalidHeader,
    #[error("invalid magic number")]
    InvalidMagic,
    #[error("invalid GGUF version: {0}")]
    InvalidVersion(i32),
    #[error("tensor count exceeds maximum ({max}): {count}")]
    TensorCountTooLarge { count: u64, max: u64 },
    #[error("metadata count exceeds maximum ({max}): {count}")]
    MetadataCountTooLarge { count: u64, max: u64 },
    #[error("tensor dimensions exceed maximum ({max}): {dims}")]
    DimensionsTooLarge { dims: u32, max: u32 },
    #[error("tensor shape too large (product overflow)")]
    ShapeTooLarge,
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

const GGUF_MAGIC: u32 = 0x46554747;
const MAX_TENSOR_COUNT: u64 = 100_000;
const MAX_METADATA_COUNT: u64 = 10_000;
const MAX_DIMENSIONS: u32 = 32;

pub fn parse_gguf<R: Read + Seek>(reader: &mut R) -> Result<Artifact, GGUFParserError> {
    let magic = read_u32(reader)?;
    if magic != GGUF_MAGIC {
        return Err(GGUFParserError::InvalidMagic);
    }

    let version = read_u32(reader)?;
    if version > 4 {
        return Err(GGUFParserError::InvalidVersion(version as i32));
    }

    let tensor_count = read_u64(reader)?;
    if tensor_count > MAX_TENSOR_COUNT {
        return Err(GGUFParserError::TensorCountTooLarge {
            count: tensor_count,
            max: MAX_TENSOR_COUNT,
        });
    }
    let metadata_kv_count = read_u64(reader)?;
    if metadata_kv_count > MAX_METADATA_COUNT {
        return Err(GGUFParserError::MetadataCountTooLarge {
            count: metadata_kv_count,
            max: MAX_METADATA_COUNT,
        });
    }

    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_kv_count {
        let (key, value) = read_kv(reader, version)?;
        metadata.insert(key, value);
    }

    let mut tensors = BTreeMap::new();
    for _ in 0..tensor_count {
        let name = read_string(reader)?;
        let n_dims = read_u32(reader)?;
        if n_dims > MAX_DIMENSIONS {
            return Err(GGUFParserError::DimensionsTooLarge {
                dims: n_dims,
                max: MAX_DIMENSIONS,
            });
        }
        let mut shape = Vec::new();
        for _ in 0..n_dims {
            shape.push(read_u64(reader)?);
        }
        let dtype = read_u32(reader)?;
        let _offset = read_u64(reader)?;
        let byte_length =
            compute_byte_length(&shape, dtype).ok_or(GGUFParserError::ShapeTooLarge)?;

        tensors.insert(
            name.clone(),
            Tensor {
                name,
                dtype: gguf_dtype_str(dtype),
                shape,
                byte_length,
            },
        );
    }

    Ok(Artifact {
        format: Format::GGUF,
        gguf_version: Some(version as i64),
        metadata,
        tensors,
    })
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32, GGUFParserError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64, GGUFParserError> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_string<R: Read>(reader: &mut R) -> Result<String, GGUFParserError> {
    let len = read_u64(reader)? as usize;
    if len == 0 {
        return Ok(String::new());
    }
    if len > 1_000_000 {
        return Err(GGUFParserError::InvalidHeader);
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|_| GGUFParserError::InvalidHeader)
}

fn read_kv<R: Read>(
    reader: &mut R,
    _version: u32,
) -> Result<(String, CanonicalValue), GGUFParserError> {
    let key = read_string(reader)?;
    let value_type = read_u32(reader)?;
    let value = match value_type {
        0 => CanonicalValue::Uint8(read_u8(reader)? as i64),
        1 => CanonicalValue::Int8(read_i8(reader)? as i64),
        2 => CanonicalValue::Uint16(read_u16(reader)? as i64),
        3 => CanonicalValue::Int16(read_i16(reader)? as i64),
        4 => CanonicalValue::Uint32(read_u32(reader)? as i64),
        5 => CanonicalValue::Int32(read_i32(reader)? as i64),
        6 => CanonicalValue::Float32(read_f32(reader)?.into()),
        7 => CanonicalValue::Bool(read_bool(reader)?),
        8 => CanonicalValue::String(read_string(reader)?),
        9 => {
            let element_type = read_u32(reader)?;
            let n = read_u64(reader)? as usize;
            let mut arr = Vec::new();
            for _ in 0..n {
                let val = match element_type {
                    0 => CanonicalValue::Uint8(read_u8(reader)? as i64),
                    1 => CanonicalValue::Int8(read_i8(reader)? as i64),
                    2 => CanonicalValue::Uint16(read_u16(reader)? as i64),
                    3 => CanonicalValue::Int16(read_i16(reader)? as i64),
                    4 => CanonicalValue::Uint32(read_u32(reader)? as i64),
                    5 => CanonicalValue::Int32(read_i32(reader)? as i64),
                    6 => CanonicalValue::Float32(read_f32(reader)?.into()),
                    7 => CanonicalValue::Bool(read_bool(reader)?),
                    8 => CanonicalValue::String(read_string(reader)?),
                    _ => return Err(GGUFParserError::InvalidHeader),
                };
                arr.push(val);
            }
            CanonicalValue::Array(arr)
        }
        10 => CanonicalValue::Uint64(read_u64(reader)? as i64),
        11 => CanonicalValue::Int64(read_i64(reader)?),
        12 => CanonicalValue::Float(read_f64(reader)?),
        _ => return Err(GGUFParserError::InvalidHeader),
    };
    Ok((key, value))
}

fn read_u8<R: Read>(reader: &mut R) -> Result<u8, GGUFParserError> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i8<R: Read>(reader: &mut R) -> Result<i8, GGUFParserError> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0] as i8)
}

fn read_u16<R: Read>(reader: &mut R) -> Result<u16, GGUFParserError> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16<R: Read>(reader: &mut R) -> Result<i16, GGUFParserError> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

fn read_i32<R: Read>(reader: &mut R) -> Result<i32, GGUFParserError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_i64<R: Read>(reader: &mut R) -> Result<i64, GGUFParserError> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32<R: Read>(reader: &mut R) -> Result<f32, GGUFParserError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64<R: Read>(reader: &mut R) -> Result<f64, GGUFParserError> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_bool<R: Read>(reader: &mut R) -> Result<bool, GGUFParserError> {
    let val = read_u8(reader)?;
    Ok(val != 0)
}

fn gguf_dtype_str(dtype: u32) -> String {
    match dtype {
        0 => "f32".to_string(),
        1 => "f16".to_string(),
        2 => "q4_0".to_string(),
        3 => "q4_1".to_string(),
        6 => "q5_0".to_string(),
        7 => "q5_1".to_string(),
        8 => "q8_0".to_string(),
        9 => "q8_1".to_string(),
        10 => "q2_k".to_string(),
        11 => "q3_k".to_string(),
        12 => "q4_k".to_string(),
        13 => "q5_k".to_string(),
        14 => "q6_k".to_string(),
        15 => "q8_k".to_string(),
        16 => "iq2_xxs".to_string(),
        17 => "iq2_xs".to_string(),
        18 => "iq3_xxs".to_string(),
        19 => "iq1_s".to_string(),
        20 => "iq4_nl".to_string(),
        21 => "iq3_s".to_string(),
        22 => "iq2_s".to_string(),
        23 => "iq4_xs".to_string(),
        24 => "i8".to_string(),
        25 => "i16".to_string(),
        26 => "i32".to_string(),
        27 => "i64".to_string(),
        28 => "f64".to_string(),
        29 => "iq1_m".to_string(),
        30 => "bf16".to_string(),
        34 => "tq1_0".to_string(),
        35 => "tq2_0".to_string(),
        39 => "mxfp4".to_string(),
        _ => format!("unknown_{}", dtype),
    }
}

fn compute_byte_length(shape: &[u64], dtype: u32) -> Option<u64> {
    let mut elements: u64 = 1;
    for &dim in shape {
        elements = elements.checked_mul(dim)?;
    }
    let byte_size = match dtype {
        0 | 26 => elements.checked_mul(4)?,      // f32, i32
        1 | 25 | 30 => elements.checked_mul(2)?, // f16, i16, bf16
        24 => elements,                          // i8
        27 | 28 => elements.checked_mul(8)?,     // i64, f64
        _ => elements,                           // Quantized types - use element count
    };
    Some(byte_size)
}
