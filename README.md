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
weight-inspect id model.gguf        # Get a stable fingerprint
weight-inspect inspect model.gguf   # See full structure details
weight-inspect diff a.gguf b.gguf   # Compare two models
weight-inspect summary model.gguf   # One-line summary for scripts
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
Structural identity
──────────────────
ID:        wi:gguf:v3:9c1f3d2a
Stable:    yes (machine independent)
Includes:  header, tensor names, shapes, dtypes
Excludes:  raw weight bytes

format: GGUF
structural_hash: 9c1f3d2a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1
tensor_count: 291
metadata_count: 12
```

### One-line summary (for scripting/CI)

```bash
$ weight-inspect summary model.gguf
gguf,3,291,12,9c1f3d2a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1
```

`summary` is designed to fit on a single line and be safe for scripting, CI pipelines, and automation.

### Inspect full details

```bash
$ weight-inspect inspect model.gguf --verbose
gguf
───
Version:  3
Tensors:  291
Metadata: 12

Structural summary
──────────────────
Dtypes:  f16 (92%), q4_k (8%)

Structural ID
─────────────
Hash: 9c1f3d2a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1

Tensors (first 10)
──────────────────
Name                                    Dtype     Shape               Bytes
────────────────────────────────────────────────────────────────────────────────
token_embd.weight                      f32       [32000, 4096]      524288000
blk.0.attn.wq.weight                  f16       [4096, 4096]       33554432
blk.0.attn.wk.weight                  f16       [4096, 4096]       33554432
blk.0.attn.wv.weight                  f16       [4096, 4096]       33554432
blk.0.attn.wo.weight                  f16       [4096, 4096]       33554432
blk.0.mlp.gate_proj.weight            f16       [11008, 4096]      90596966
...
```

### Compare two files

```bash
$ weight-inspect diff a.gguf b.gguf
DIFFERENT
--------------------
Added tensors:    2
Removed tensors:  0
Modified tensors: 3
Structural ID:    changed

Tensor changes
─────────────
  blk.12.attn.wq.weight
    dtype: f16 -> q4_k
    shape: [4096, 4096] -> [4096, 5120]
  blk.18.mlp.up_proj
    dtype: f16 -> q4_k
  output.weight
    shape: [32000, 4096] -> [32000, 5120]
```

Or for PR comments:

```bash
$ weight-inspect diff a.gguf b.gguf --format md
## Structural Diff

**Status:** ❌ DIFFERENT

| Change Type | Count |
|-------------|-------|
| Added tensors | 2 |
| Removed tensors | 0 |
| Modified tensors | 3 |
```

### Compare with exit codes (CI)

```bash
# Exit code 0 if identical, 1 if different
weight-inspect diff a.gguf b.gguf --fail-on-diff
```

### Show only changes

```bash
weight-inspect diff a.gguf b.gguf --only-changes
```

### Verbose output

```bash
weight-inspect inspect model.gguf --verbose
weight-inspect diff a.gguf b.gguf --verbose
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

## Roadmap

- [ ] Support for more formats (PyTorch .pt, TensorFlow .pb)
- [ ] Configurable hash inputs (include/exclude metadata)
- [ ] Output format plugins
- [ ] Git-like status porcelain

## Stability Guarantees

- **Output ordering**: Alphabetically sorted (BTreeMap)
- **Hash versioning**: Schema field in JSON output for future compatibility
- **Breaking changes**: Will bump major version in CLI and JSON schema
- **CI-safe**: Use `--fail-on-diff` for exit codes

## License

MIT
