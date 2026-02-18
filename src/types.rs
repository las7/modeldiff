use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Artifact {
    pub format: Format,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gguf_version: Option<i64>,
    pub metadata: BTreeMap<String, CanonicalValue>,
    pub tensors: BTreeMap<String, Tensor>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    GGUF,
    Safetensors,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CanonicalValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Array(Vec<CanonicalValue>),
    Uint8(i64),
    Int8(i64),
    Uint16(i64),
    Int16(i64),
    Uint32(i64),
    Int32(i64),
    Uint64(i64),
    Int64(i64),
    Float32(f64),
}

pub struct CanonicalSerializer;

impl CanonicalSerializer {
    pub fn serialize_value(&self, value: &CanonicalValue) -> String {
        match value {
            CanonicalValue::Null => "null".to_string(),
            CanonicalValue::Bool(b) => b.to_string(),
            CanonicalValue::Int(i) => i.to_string(),
            CanonicalValue::Float(fl) => fl.to_bits().to_string(),
            CanonicalValue::String(s) => format!("\"{}\"", escape_string(s)),
            CanonicalValue::Array(arr) => {
                let items: Vec<String> = arr.iter().map(|v| self.serialize_value(v)).collect();
                format!("[{}]", items.join(","))
            }
            CanonicalValue::Uint8(i) => (*i).to_string(),
            CanonicalValue::Int8(i) => (*i).to_string(),
            CanonicalValue::Uint16(i) => (*i).to_string(),
            CanonicalValue::Int16(i) => (*i).to_string(),
            CanonicalValue::Uint32(i) => (*i).to_string(),
            CanonicalValue::Int32(i) => (*i).to_string(),
            CanonicalValue::Uint64(i) => (*i).to_string(),
            CanonicalValue::Int64(i) => (*i).to_string(),
            CanonicalValue::Float32(fl) => format!("f32:{}", (*fl).to_bits()),
        }
    }
}

fn escape_string(s: &str) -> String {
    let mut result = String::new();
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c.is_ascii_graphic() || c == ' ' => result.push(c),
            _ => result.push_str(&format!("\\u{:04x}", c as u32)),
        }
    }
    result
}

impl Serialize for CanonicalValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let s = CanonicalSerializer.serialize_value(self);
        serializer.serialize_str(&s)
    }
}

impl<'de> Deserialize<'de> for CanonicalValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;

        if s == "null" {
            return Ok(CanonicalValue::Null);
        }
        if s == "true" {
            return Ok(CanonicalValue::Bool(true));
        }
        if s == "false" {
            return Ok(CanonicalValue::Bool(false));
        }

        if let Ok(i) = s.parse::<i64>() {
            return Ok(CanonicalValue::Int(i));
        }

        if let Some(bits) = s.strip_prefix("f32:")
            && let Ok(bits) = bits.parse::<u32>()
        {
            return Ok(CanonicalValue::Float32(f32::from_bits(bits).into()));
        }

        if let Ok(bits) = s.parse::<u64>() {
            let fl = f64::from_bits(bits);
            return Ok(CanonicalValue::Float(fl));
        }

        if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\\') && s.contains(':')) {
            return Ok(CanonicalValue::String(s));
        }

        Ok(CanonicalValue::String(s))
    }
}

impl fmt::Display for CanonicalValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CanonicalValue::Null => write!(f, "null"),
            CanonicalValue::Bool(b) => write!(f, "{}", b),
            CanonicalValue::Int(i) => write!(f, "{}", i),
            CanonicalValue::Float(fl) => write!(f, "{}", fl),
            CanonicalValue::String(s) => write!(f, "{}", s),
            CanonicalValue::Array(arr) => write!(f, "{:?}", arr),
            CanonicalValue::Uint8(i) => write!(f, "{}", i),
            CanonicalValue::Int8(i) => write!(f, "{}", i),
            CanonicalValue::Uint16(i) => write!(f, "{}", i),
            CanonicalValue::Int16(i) => write!(f, "{}", i),
            CanonicalValue::Uint32(i) => write!(f, "{}", i),
            CanonicalValue::Int32(i) => write!(f, "{}", i),
            CanonicalValue::Uint64(i) => write!(f, "{}", i),
            CanonicalValue::Int64(i) => write!(f, "{}", i),
            CanonicalValue::Float32(fl) => write!(f, "{}", fl),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub byte_length: u64,
}
