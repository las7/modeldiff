use crate::types::{Artifact, CanonicalValue};
use serde::Serialize;
use std::collections::BTreeSet;

#[derive(Debug, Default, Serialize)]
pub struct DiffResult {
    pub schema: u32,
    pub format_equal: bool,
    pub hash_equal: bool,
    pub tensor_count_equal: bool,
    pub metadata_count_equal: bool,
    pub metadata_added: Vec<String>,
    pub metadata_removed: Vec<String>,
    pub metadata_changed: Vec<MetadataChange>,
    pub tensors_added: Vec<String>,
    pub tensors_removed: Vec<String>,
    pub tensor_changes: Vec<TensorChange>,
}

impl DiffResult {
    pub fn new() -> Self {
        Self {
            schema: 1,
            ..Default::default()
        }
    }
}

#[derive(Debug, Serialize)]
pub struct MetadataChange {
    pub key: String,
    pub old_value: CanonicalValue,
    pub new_value: CanonicalValue,
}

#[derive(Debug, Serialize)]
pub struct TensorChange {
    pub name: String,
    pub dtype_old: Option<String>,
    pub dtype_new: Option<String>,
    pub shape_old: Option<Vec<u64>>,
    pub shape_new: Option<Vec<u64>>,
    pub byte_length_old: Option<u64>,
    pub byte_length_new: Option<u64>,
}

pub fn diff(a: &Artifact, b: &Artifact) -> DiffResult {
    let mut result = DiffResult::new();
    result.format_equal = a.format == b.format;
    result.tensor_count_equal = a.tensors.len() == b.tensors.len();
    result.metadata_count_equal = a.metadata.len() == b.metadata.len();

    let keys_a: BTreeSet<_> = a.metadata.keys().collect();
    let keys_b: BTreeSet<_> = b.metadata.keys().collect();

    for key in keys_b.difference(&keys_a) {
        result.metadata_added.push((*key).clone());
    }
    for key in keys_a.difference(&keys_b) {
        result.metadata_removed.push((*key).clone());
    }
    for key in keys_a.intersection(&keys_b) {
        let old_val = a.metadata.get(*key).unwrap();
        let new_val = b.metadata.get(*key).unwrap();
        if old_val != new_val {
            result.metadata_changed.push(MetadataChange {
                key: (*key).clone(),
                old_value: old_val.clone(),
                new_value: new_val.clone(),
            });
        }
    }

    let tensor_names_a: BTreeSet<_> = a.tensors.keys().collect();
    let tensor_names_b: BTreeSet<_> = b.tensors.keys().collect();

    for name in tensor_names_b.difference(&tensor_names_a) {
        result.tensors_added.push((*name).clone());
    }
    for name in tensor_names_a.difference(&tensor_names_b) {
        result.tensors_removed.push((*name).clone());
    }
    for name in tensor_names_a.intersection(&tensor_names_b) {
        let old_tensor = a.tensors.get(*name).unwrap();
        let new_tensor = b.tensors.get(*name).unwrap();

        let mut change = TensorChange {
            name: (*name).clone(),
            dtype_old: None,
            dtype_new: None,
            shape_old: None,
            shape_new: None,
            byte_length_old: None,
            byte_length_new: None,
        };

        if old_tensor.dtype != new_tensor.dtype {
            change.dtype_old = Some(old_tensor.dtype.clone());
            change.dtype_new = Some(new_tensor.dtype.clone());
        }
        if old_tensor.shape != new_tensor.shape {
            change.shape_old = Some(old_tensor.shape.clone());
            change.shape_new = Some(new_tensor.shape.clone());
        }
        if old_tensor.byte_length != new_tensor.byte_length {
            change.byte_length_old = Some(old_tensor.byte_length);
            change.byte_length_new = Some(new_tensor.byte_length);
        }

        if change.dtype_old.is_some()
            || change.shape_old.is_some()
            || change.byte_length_old.is_some()
        {
            result.tensor_changes.push(change);
        }
    }

    result
}

impl DiffResult {
    pub fn has_changes(&self) -> bool {
        !self.metadata_added.is_empty()
            || !self.metadata_removed.is_empty()
            || !self.metadata_changed.is_empty()
            || !self.tensors_added.is_empty()
            || !self.tensors_removed.is_empty()
            || !self.tensor_changes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::diff;
    use crate::types::{Artifact, CanonicalValue, Format, Tensor};
    use std::collections::BTreeMap;

    fn create_test_artifact(
        format: Format,
        metadata_count: usize,
        tensor_count: usize,
    ) -> Artifact {
        let mut metadata = BTreeMap::new();
        for i in 0..metadata_count {
            metadata.insert(
                format!("key_{}", i),
                CanonicalValue::String(format!("value_{}", i)),
            );
        }

        let mut tensors = BTreeMap::new();
        for i in 0..tensor_count {
            tensors.insert(
                format!("tensor_{}", i),
                Tensor {
                    name: format!("tensor_{}", i),
                    dtype: "f32".to_string(),
                    shape: vec![10, 10],
                    byte_length: 400,
                },
            );
        }

        Artifact {
            format,
            gguf_version: Some(3),
            metadata,
            tensors,
        }
    }

    #[test]
    fn test_diff_identical_artifacts() {
        let a = create_test_artifact(Format::GGUF, 5, 3);
        let b = a.clone();

        let result = diff(&a, &b);

        assert!(result.format_equal);
        assert!(result.tensor_count_equal);
        assert!(result.metadata_count_equal);
        assert!(result.metadata_added.is_empty());
        assert!(result.metadata_removed.is_empty());
        assert!(result.metadata_changed.is_empty());
        assert!(result.tensors_added.is_empty());
        assert!(result.tensors_removed.is_empty());
        assert!(result.tensor_changes.is_empty());
    }

    #[test]
    fn test_diff_format_mismatch() {
        let a = create_test_artifact(Format::GGUF, 5, 3);
        let b = create_test_artifact(Format::Safetensors, 5, 3);

        let result = diff(&a, &b);

        assert!(!result.format_equal);
    }

    #[test]
    fn test_diff_metadata_added() {
        let a = create_test_artifact(Format::GGUF, 3, 0);
        let mut b = a.clone();
        b.metadata.insert(
            "new_key".to_string(),
            CanonicalValue::String("new_value".to_string()),
        );

        let result = diff(&a, &b);

        assert!(result.metadata_added.contains(&"new_key".to_string()));
    }

    #[test]
    fn test_diff_metadata_removed() {
        let a = create_test_artifact(Format::GGUF, 3, 0);
        let mut b = a.clone();
        b.metadata.remove("key_0");

        let result = diff(&a, &b);

        assert!(result.metadata_removed.contains(&"key_0".to_string()));
    }

    #[test]
    fn test_diff_metadata_changed() {
        let mut a = create_test_artifact(Format::GGUF, 3, 0);
        let mut b = a.clone();
        a.metadata.insert(
            "key_0".to_string(),
            CanonicalValue::String("old_value".to_string()),
        );
        b.metadata.insert(
            "key_0".to_string(),
            CanonicalValue::String("new_value".to_string()),
        );

        let result = diff(&a, &b);

        assert_eq!(result.metadata_changed.len(), 1);
        assert_eq!(result.metadata_changed[0].key, "key_0");
    }

    #[test]
    fn test_diff_tensor_added() {
        let a = create_test_artifact(Format::GGUF, 0, 2);
        let mut b = a.clone();
        b.tensors.insert(
            "new_tensor".to_string(),
            Tensor {
                name: "new_tensor".to_string(),
                dtype: "f32".to_string(),
                shape: vec![10],
                byte_length: 40,
            },
        );

        let result = diff(&a, &b);

        assert!(result.tensors_added.contains(&"new_tensor".to_string()));
    }

    #[test]
    fn test_diff_tensor_shape_changed() {
        let a = create_test_artifact(Format::GGUF, 0, 2);
        let mut b = a.clone();
        b.tensors.get_mut("tensor_0").unwrap().shape = vec![20, 20];

        let result = diff(&a, &b);

        assert_eq!(result.tensor_changes.len(), 1);
        assert_eq!(result.tensor_changes[0].name, "tensor_0");
        assert!(result.tensor_changes[0].shape_old.is_some());
    }

    #[test]
    fn test_diff_tensor_dtype_changed() {
        let a = create_test_artifact(Format::GGUF, 0, 2);
        let mut b = a.clone();
        b.tensors.get_mut("tensor_0").unwrap().dtype = "f16".to_string();

        let result = diff(&a, &b);

        assert_eq!(result.tensor_changes.len(), 1);
        assert_eq!(result.tensor_changes[0].dtype_old, Some("f32".to_string()));
        assert_eq!(result.tensor_changes[0].dtype_new, Some("f16".to_string()));
    }

    #[test]
    fn test_diff_has_changes() {
        let a = create_test_artifact(Format::GGUF, 3, 2);
        let b = create_test_artifact(Format::GGUF, 5, 4);

        let result = diff(&a, &b);

        assert!(result.has_changes());
    }

    #[test]
    fn test_diff_no_changes() {
        let a = create_test_artifact(Format::GGUF, 3, 2);
        let b = a.clone();

        let result = diff(&a, &b);

        assert!(!result.has_changes());
    }

    #[test]
    fn test_determinism_metadata_order() {
        use crate::hash::compute_structural_hash;

        let mut a = create_test_artifact(Format::GGUF, 3, 0);
        let mut b = create_test_artifact(Format::GGUF, 3, 0);

        a.metadata.insert(
            "zzz_key".to_string(),
            CanonicalValue::String("zzz".to_string()),
        );
        a.metadata.insert(
            "aaa_key".to_string(),
            CanonicalValue::String("aaa".to_string()),
        );

        b.metadata.insert(
            "aaa_key".to_string(),
            CanonicalValue::String("aaa".to_string()),
        );
        b.metadata.insert(
            "zzz_key".to_string(),
            CanonicalValue::String("zzz".to_string()),
        );

        let hash_a = compute_structural_hash(&a).unwrap();
        let hash_b = compute_structural_hash(&b).unwrap();

        assert_eq!(hash_a, hash_b, "Metadata order should not affect hash");
    }

    #[test]
    fn test_determinism_tensor_order() {
        use crate::hash::compute_structural_hash;

        let mut a = create_test_artifact(Format::GGUF, 0, 2);
        let mut b = create_test_artifact(Format::GGUF, 0, 2);

        a.tensors.insert(
            "zzz_tensor".to_string(),
            Tensor {
                name: "zzz_tensor".to_string(),
                dtype: "f32".to_string(),
                shape: vec![10],
                byte_length: 40,
            },
        );
        a.tensors.insert(
            "aaa_tensor".to_string(),
            Tensor {
                name: "aaa_tensor".to_string(),
                dtype: "f32".to_string(),
                shape: vec![10],
                byte_length: 40,
            },
        );

        b.tensors.insert(
            "aaa_tensor".to_string(),
            Tensor {
                name: "aaa_tensor".to_string(),
                dtype: "f32".to_string(),
                shape: vec![10],
                byte_length: 40,
            },
        );
        b.tensors.insert(
            "zzz_tensor".to_string(),
            Tensor {
                name: "zzz_tensor".to_string(),
                dtype: "f32".to_string(),
                shape: vec![10],
                byte_length: 40,
            },
        );

        let hash_a = compute_structural_hash(&a).unwrap();
        let hash_b = compute_structural_hash(&b).unwrap();

        assert_eq!(hash_a, hash_b, "Tensor order should not affect hash");
    }
}
