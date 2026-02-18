# modeldiff

Deterministic structural identity for GGUF and safetensors ML model files.

## The Problem

Model files are opaque blobs. When you download or convert a model, you can't easily answer:

- "What model family/architecture is this?"
- "What quantization is used?"
- "What are the tensor shapes?"
- "Is this the same model as before?"

## The Solution

modeldiff provides **structural identity** - a canonical, deterministic representation of what a model file actually contains.

### Core Value

- **Understand** what any model file contains
- **Compare** models structurally
- **Reproduce** identical results across machines

## Features

- **Inspect** - Understand any model file's structure
- **Hash** - Compute deterministic structural identity (SHA256 of canonical JSON)
- **Diff** - Compare two models structurally (built on identity)

## Installation

```bash
cargo install modeldiff
```

Or build from source:

```bash
cargo build --release
```

## Usage

```bash
# Inspect a model file
modeldiff inspect model.gguf

# Inspect with JSON output
modeldiff inspect model.gguf --json

# Get structural identity
modeldiff id model.gguf

# ID with JSON output
modeldiff id model.gguf --json

# Diff two files
modeldiff diff a.gguf b.gguf

# JSON output (for tooling/integration)
modeldiff diff a.gguf b.gguf --json

# Quiet mode (exit 0 if identical, 1 if different)
modeldiff diff a.gguf b.gguf --quiet
```

## What it does

- Parses GGUF and safetensors headers
- Extracts metadata (hyperparameters, tokenizer config, etc.)
- Lists tensor names, dtypes, and shapes
- Computes structural hash (deterministic JSON → SHA256)
- Compares two files showing:
  - Added/removed/changed metadata keys
  - Added/removed/changed tensors
  - Tensor dtype/shape/byte_length changes

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

- "Why are these two model files different?"
- Verify model file integrity
- Compare quantization variants
- Detect structural changes in converted models

## License

MIT
