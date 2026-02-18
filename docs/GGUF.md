# GGUF Format Guide

GGUF (GGML Universal Format) is a binary format for storing ML models. This guide explains the structure for understanding model files.

## File Structure

```
┌─────────────────────────────────────┐
│ Header (metadata + tensor count)   │
├─────────────────────────────────────┤
│ Tensor Info (names, shapes, types) │
├─────────────────────────────────────┤
│ Padding (to alignment boundary)     │
├─────────────────────────────────────┤
│ Tensor Data (weights)              │
└─────────────────────────────────────┘
```

## Header

| Field | Type | Description |
|-------|------|-------------|
| magic | u32 | "GGUF" (0x46554747) |
| version | u32 | Format version (currently 3) |
| tensor_count | u64 | Number of tensors |
| metadata_kv_count | u64 | Number of metadata key-value pairs |
| metadata_kv[] | KV pairs | Model hyperparameters |

## Metadata Key-Value Pairs

Metadata stores model configuration. Keys are hierarchical (e.g., `llama.attention.head_count`).

### Common Keys

**General:**
- `general.architecture` - Model architecture (llama, mamba, gpt2, etc.)
- `general.name` - Model name
- `general.quantization_version` - Quantization format version

**LLM Common:**
- `[arch].context_length` - Maximum context size
- `[arch].embedding_length` - Embedding dimension
- `[arch].block_count` - Number of layers
- `[arch].feed_forward_length` - FFN dimension
- `[arch].attention.head_count` - Number of attention heads

**Tokenizer:**
- `tokenizer.ggml.model` - Tokenizer type (llama, gpt2, etc.)
- `tokenizer.ggml.tokens` - Vocabulary array
- `tokenizer.ggml.merges` - BPE merges (for BPE tokenizers)

**Quantization:**
- `general.file_type` - Quantization type (1=F16, 15=Q4_K_M, etc.)

## Data Types

GGUF supports various tensor types:

| Type ID | Name | Description |
|---------|------|-------------|
| 0 | F32 | 32-bit float |
| 1 | F16 | 16-bit float |
| 2 | Q4_0 | 4-bit quantization |
| 3 | Q4_1 | 4-bit quantization |
| 10 | Q2_K | 2-bit K-quant |
| 11 | Q3_K | 3-bit K-quant |
| 12 | Q4_K | 4-bit K-quant |
| 13 | Q5_K | 5-bit K-quant |
| 14 | Q6_K | 6-bit K-quant |
| 24 | I8 | 8-bit integer |
| 25 | I16 | 16-bit integer |
| 26 | I32 | 32-bit integer |
| 30 | BF16 | Brain float16 |

## Tensor Info

Each tensor has:

- **name**: String identifier (e.g., `blk.0.attn_q.weight`)
- **n_dimensions**: Number of dimensions (1-4)
- **dimensions[]**: Shape of each dimension
- **type**: Data type (see above)
- **offset**: Byte offset in file (for loading data)
- **byte_length**: Size in bytes

## Tensor Naming Convention

Tensors follow a hierarchical naming:

```
blk.{layer}.{component}.{subcomponent}.{type}
```

Examples:
- `token_embd.weight` - Token embeddings
- `blk.0.attn_q.weight` - Attention query weights
- `blk.0.attn_k.weight` - Attention key weights
- `blk.0.attn_v.weight` - Attention value weights
- `blk.0.attn_output.weight` - Attention output projection
- `blk.0.ffn_gate.weight` - FFN gate (SwiGLU)
- `blk.0.ffn_down.weight` - FFN down projection
- `output.weight` - Output layer

## What weight-inspect Reads

weight-inspect only reads:

1. **Header** - Magic, version, counts
2. **Metadata** - All key-value pairs
3. **Tensor Info** - Names, shapes, types, byte lengths

It does NOT read:
- Tensor data/weights
- File offsets (for data loading)

This is intentional for fast, safe structural analysis.

## Example: Inspecting a GGUF File

```bash
weight-inspect inspect model.gguf
```

Output:
```
format: GGUF
gguf_version: 3
tensor_count: 242
metadata_count: 22
structural_hash: abc123...

First 5 tensors:
  1: blk.0.attn_norm.weight [768] (f32)
  2: blk.0.ssm_a [16, 1536] (f32)
  ...
```

## References

- [GGUF Specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [llama.cpp GGUF Documentation](https://github.com/ggerganov/llama.cpp/wiki/GGUF)
