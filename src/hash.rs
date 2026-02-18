use crate::types::Artifact;
use sha2::{Digest, Sha256};

pub fn compute_structural_hash(artifact: &Artifact) -> Result<String, serde_json::Error> {
    let canonical = serde_json::to_string(artifact)?;
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    let result = hasher.finalize();
    Ok(hex::encode(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Format, Tensor};
    use std::collections::BTreeMap;

    #[test]
    fn test_hash_determinism() {
        let mut artifact1 = Artifact {
            format: Format::GGUF,
            gguf_version: Some(3),
            metadata: BTreeMap::new(),
            tensors: BTreeMap::new(),
        };
        artifact1.metadata.insert(
            "test".to_string(),
            crate::types::CanonicalValue::String("value".to_string()),
        );

        let hash1 = compute_structural_hash(&artifact1).unwrap();

        let mut artifact2 = Artifact {
            format: Format::GGUF,
            gguf_version: Some(3),
            metadata: BTreeMap::new(),
            tensors: BTreeMap::new(),
        };
        artifact2.metadata.insert(
            "test".to_string(),
            crate::types::CanonicalValue::String("value".to_string()),
        );

        let hash2 = compute_structural_hash(&artifact2).unwrap();

        assert_eq!(hash1, hash2, "same artifact should produce same hash");
    }

    #[test]
    fn test_hash_different_artifacts_different_hashes() {
        let mut artifact1 = Artifact {
            format: Format::GGUF,
            gguf_version: Some(3),
            metadata: BTreeMap::new(),
            tensors: BTreeMap::new(),
        };
        artifact1.metadata.insert(
            "test".to_string(),
            crate::types::CanonicalValue::String("value1".to_string()),
        );

        let mut artifact2 = Artifact {
            format: Format::GGUF,
            gguf_version: Some(3),
            metadata: BTreeMap::new(),
            tensors: BTreeMap::new(),
        };
        artifact2.metadata.insert(
            "test".to_string(),
            crate::types::CanonicalValue::String("value2".to_string()),
        );

        let hash1 = compute_structural_hash(&artifact1).unwrap();
        let hash2 = compute_structural_hash(&artifact2).unwrap();

        assert_ne!(
            hash1, hash2,
            "different artifacts should produce different hashes"
        );
    }

    #[test]
    fn test_hash_format_affects_hash() {
        let artifact1 = Artifact {
            format: Format::GGUF,
            gguf_version: Some(3),
            metadata: BTreeMap::new(),
            tensors: BTreeMap::new(),
        };

        let artifact2 = Artifact {
            format: Format::Safetensors,
            gguf_version: None,
            metadata: BTreeMap::new(),
            tensors: BTreeMap::new(),
        };

        let hash1 = compute_structural_hash(&artifact1).unwrap();
        let hash2 = compute_structural_hash(&artifact2).unwrap();

        assert_ne!(
            hash1, hash2,
            "different formats should produce different hashes"
        );
    }

    #[test]
    fn test_hash_tensor_count_affects_hash() {
        let mut artifact1 = Artifact {
            format: Format::GGUF,
            gguf_version: Some(3),
            metadata: BTreeMap::new(),
            tensors: BTreeMap::new(),
        };
        artifact1.tensors.insert(
            "tensor1".to_string(),
            Tensor {
                name: "tensor1".to_string(),
                dtype: "f32".to_string(),
                shape: vec![10],
                byte_length: 40,
            },
        );

        let mut artifact2 = Artifact {
            format: Format::GGUF,
            gguf_version: Some(3),
            metadata: BTreeMap::new(),
            tensors: BTreeMap::new(),
        };
        artifact2.tensors.insert(
            "tensor1".to_string(),
            Tensor {
                name: "tensor1".to_string(),
                dtype: "f32".to_string(),
                shape: vec![10],
                byte_length: 40,
            },
        );
        artifact2.tensors.insert(
            "tensor2".to_string(),
            Tensor {
                name: "tensor2".to_string(),
                dtype: "f32".to_string(),
                shape: vec![10],
                byte_length: 40,
            },
        );

        let hash1 = compute_structural_hash(&artifact1).unwrap();
        let hash2 = compute_structural_hash(&artifact2).unwrap();

        assert_ne!(
            hash1, hash2,
            "different tensor counts should produce different hashes"
        );
    }
}
