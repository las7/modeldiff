# weight-inspect

Inspect GGUF and safetensors model files to see what's inside.

## The Problem

Model files are opaque blobs. When you download or convert a model, you can't easily answer:

- "What model architecture is this?"
- "What quantization is used?"
- "What are the tensor shapes?"
- "Is this the same model as before?"

## The Solution

weight-inspect reads model file headers and gives you a **fingerprint** - a unique hash based on the model's structure (not the actual weights).

This lets you:
- See what's in any model file
- Compare two models
- Verify if models are structurally the same

## Quick Start

```bash
# Get a fingerprint (the core operation)
weight-inspect id model.gguf

# Inspect full details
weight-inspect inspect model.gguf

# Compare two models
weight-inspect diff a.gguf b.gguf

# One-line summary (for GitHub issues)
weight-inspect summary model.gguf
```

## Installation

```bash
cargo install weight-inspect
```

Or build from source:

```bash
cargo build --release
```

## Usage Examples

### Get a fingerprint

```bash
$ weight-inspect id model.gguf
format: GGUF
structural_hash: 2de46f849506b6a34b02d75796396ab6e1b9e24844a732de2a498463aa98376d
tensor_count: 0
metadata_count: 22
```

### One-line summary (for GitHub issues)

```bash
$ weight-inspect summary model.gguf
gguf,3,0,22,2de46f849506b6a34b02d75796396ab6e1b9e24844a732de2a498463aa98376d
```

### Inspect full details

```bash
$ weight-inspect inspect model.gguf
format: GGUF
gguf_version: 3
tensor_count: 242
metadata_count: 22
structural_hash: a0c216cce9dec82633dc61eb21c96b8e66609be26aee149caea9f2eb885409e0

First 5 tensors:
  1: blk.0.attn_norm.weight [768] (f32)
  2: blk.0.ssm_a [16, 1536] (f32)
  3: blk.0.ssm_conv1d.bias [1536] (f32)
  4: blk.0.ssm_conv1d.weight [4, 1536] (f32)
  5: blk.0.ssm_d [1536] (f32)
  ... and 237 more
```

### Compare two files

```bash
$ weight-inspect diff a.gguf b.gguf
Structural Identity:
  format equal: true
  hash equal: false
  tensor count equal: true
  metadata count equal: false

Metadata:
  + gpt2.attention.head_count
  - llama.attention.head_count
  ~ general.architecture: llama -> gpt2

Tensors:
  ~ blk.0.attn_q.weight:
      dtype: f16 -> q4_k
```

### JSON output (for tooling)

```bash
$ weight-inspect id model.gguf --json
{
  "schema": 1,
  "format": "gguf",
  "structural_hash": "2de46f849506b6a34b02d75796396ab6e1b9e24844a732de2a498463aa98376d",
  "tensor_count": 0,
  "metadata_count": 22
}
```

## What it does

- Parses GGUF and safetensors headers
- Extracts metadata (hyperparameters, tokenizer config, etc.)
- Lists tensor names, dtypes, and shapes
- Computes structural hash (deterministic JSON → SHA256)
- Compares two files showing structural differences

## What it does NOT do

- **Never loads weight data** - Only reads headers and tensor descriptors
- Execute models
- Predict runtime compatibility
- Security scanning

### Why no weight data?

We intentionally only read headers/metadata because:

- **Fast**: Works on huge models without loading gigabytes
- **Memory safe**: No memory spikes on large files
- **Cross-platform**: Deterministic across any machine
- **Structural focus**: Answer "are these the same model?" not "are these identical files?"

If you need content equality (byte-for-byte), that's a separate mode that would require loading all weights.

## Design Principles

1. **Deterministic**: Same input → same output across machines
2. **Canonical**: Floats serialized as bit patterns, maps sorted
3. **No heuristics**: Pure structural comparison
4. **Fail fast**: Invalid files produce clear errors

See [SPEC.md](SPEC.md) for full specification.

## Use cases

- "What model is this?" - Get instant structural identity
- "Is this the same model?" - Compare structural hashes
- "Why are these two models different?" - See exact differences
- Verify model file integrity
- Compare quantization variants
- Detect structural changes in converted models

## License

MIT
