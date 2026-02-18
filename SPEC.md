# weight-inspect Specification

## Overview

`weight-inspect` provides deterministic structural identity for GGUF and safetensors model files.

**Scope:**
- Parse GGUF and safetensors headers/metadata
- Extract tensor descriptors (name, dtype, shape, byte_length)
- Compute structural hash
- Diff two files structurally (built on identity)

**Out of Scope:**
- Loading or hashing weight data
- Runtime compatibility prediction
- Model execution

## Structural Hash

### Definition

```
structural_hash = SHA256(canonical_json(artifact))
```

### What is Included

| Field | Included | Notes |
|-------|----------|-------|
| format | ✓ | "gguf" or "safetensors" |
| gguf_version | ✓ | Only for GGUF files |
| metadata keys | ✓ | Sorted lexicographically |
| metadata values | ✓ | Normalized (see below) |
| tensor names | ✓ | Sorted lexicographically |
| tensor dtype | ✓ | |
| tensor shape | ✓ | Sorted by dimension order |
| tensor byte_length | ✓ | |

### What is Excluded

- File offsets
- Padding
- Physical tensor ordering
- Header formatting
- Weight/tensor data bytes

## Canonicalization Rules

### 1. Map Ordering

All maps MUST be sorted by key:
- `metadata`: BTreeMap → lexicographically sorted
- `tensors`: BTreeMap → lexicographically sorted by tensor name

### 2. Number Normalization

**Integers:** Serialize as JSON integers (no trailing decimals).

**Floats:** Serialize as raw bit patterns (as JSON numbers) to ensure reproducibility:
- `Float`: bit pattern as u64 (e.g., `4611686018427387904`)
- `Float32`: prefix with `f32:` then bit pattern as u32 (e.g., `f32:1077936128`)

This prevents floating-point representation differences across platforms.

### 3. String Normalization

Strings are escaped for JSON serialization. No Unicode normalization is performed.

### 4. Array Ordering

- Metadata arrays: preserve declared order
- Tensor shapes: preserve dimension order (already ordered by spec)

## Determinism Promise

Given identical input files:
1. Parsing produces identical `Artifact` structure
2. Canonical JSON is byte-for-byte identical
3. SHA256 hash is identical
4. Diff output order is stable

**Cross-machine:** Same hash guaranteed across any machine/OS.

## Output Formats

### CLI: `weight-inspect inspect <file>`

```
format: gguf
gguf_version: 3
tensor_count: 242
metadata_count: 22
structural_hash: abc123...

First 5 tensors:
  1: blk.0.attn_norm.weight [768] (f32)
  ...
```

### CLI: `weight-inspect id <file>`

```
format: gguf
structural_hash: abc123...
tensor_count: 242
metadata_count: 22
```

### CLI: `weight-inspect diff <a> <b>`

```
Structural Identity:
  format equal: true
  hash equal: false
  tensor count equal: true
  metadata count equal: false

Metadata:
  + key_added
  - key_removed
  ~ key_changed: old -> new

Tensors:
  + new_tensor
  - removed_tensor
  ~ tensor_name:
      dtype: f16 -> q4_k
      shape: [4096] -> [4096, 4096]
```

### JSON Output

All commands support `--json`:

```json
{
  "schema": 1,
  "format": "gguf",
  "structural_hash": "abc123...",
  ...
}
```

Schema version allows downstream tools to handle format evolution.

## Error Handling

- Invalid magic bytes → "unable to parse GGUF header"
- Invalid JSON in safetensors → "invalid safetensors JSON header"
- File not found → standard IO error

No recovery logic - fail fast.

## Version History

- v0.1.0: Initial release
