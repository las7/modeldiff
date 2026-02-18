# ONNX Format Guide

ONNX (Open Neural Network Exchange) is a binary format for representing ML models. This guide explains the structure for understanding model files.

## File Structure

ONNX uses Protocol Buffers (protobuf) for serialization:

```
┌─────────────────────────────────────┐
│ ModelProto (protobuf message)       │
│  ├─ IR version, opset imports       │
│  ├─ Metadata (producer, domain)    │
│  └─ GraphProto                     │
│      ├─ Inputs/Outputs             │
│      ├─ Initializers (weights)     │
│      └─ Nodes (operations)         │
└─────────────────────────────────────┘
```

Note: ONNX does NOT have a magic byte header. Files are identified by `.onnx` extension and successful protobuf parsing.

## ModelProto

The top-level ONNX structure:

| Field | Type | Description |
|-------|------|-------------|
| ir_version | int64 | ONNX IR version (3-11) |
| opset_import | OperatorSetId[] | Required operator sets |
| producer_name | string | Tool that generated the model |
| producer_version | string | Version of generating tool |
| domain | string | Model namespace (e.g., "org.onnx") |
| model_version | int64 | Model version number |
| doc_string | string | Human-readable documentation |
| graph | GraphProto | The computation graph |
| metadata_props | map<string,string> | Custom metadata |

### IR Versions

| Version | ONNX Release | Features |
|---------|--------------|----------|
| 3 | 1.2 | Initial release |
| 4 | 1.3 | Tensor binding |
| 6 | 1.6 | Sequences, maps |
| 7 | 1.10 | Training support |
| 8 | 1.14 | Dynamic optionals |
| 11 | 1.18 | Multi-device support |

## GraphProto

The computation graph:

| Field | Type | Description |
|-------|------|-------------|
| name | string | Graph name |
| node | NodeProto[] | Computation nodes |
| initializer | TensorProto[] | Constant tensors (weights) |
| input | ValueInfoProto[] | Graph inputs |
| output | ValueInfoProto[] | Graph outputs |
| value_info | ValueInfoProto[] | Intermediate values |

## NodeProto

Each node represents an operation:

| Field | Type | Description |
|-------|------|-------------|
| name | string | Optional node name |
| input | string[] | Input tensor names |
| output | string[] | Output tensor names |
| op_type | string | Operation type (Conv, MatMul, etc.) |
| domain | string | Operator domain |
| attribute | AttributeProto[] | Operation attributes |

### Common Operator Types

**Neural Network:**
- Conv, MaxPool, AveragePool
- MatMul, Gemm
- Add, Mul, Div, Sub
- Relu, Sigmoid, Tanh, Softmax
- LSTM, GRU, RNN
- Attention, SkipLayerNorm

**Tensor:**
- Reshape, Transpose, Concat, Split
- Gather, Scatter
- Flatten, Squeeze, Unsqueeze

**Math:**
- ReduceSum, ReduceMean, ReduceMax
- ArgMax, ArgMin
- MatMul, Transpose

## TensorProto

Represents constant tensors (weights):

| Field | Type | Description |
|-------|------|-------------|
| name | string | Tensor name |
| dims | int64[] | Tensor shape |
| data_type | DataType | Element type |
| raw_data | bytes | Serialized tensor data |
| float_data | float[] | Float32 values |
| int32_data | int32[] | Int32 values |
| int64_data | int64[] | Int64 values |
| string_data | bytes[] | String values |
| external_data | string[] | External file references |

### Data Types

| Type ID | Name | Description |
|---------|------|-------------|
| 1 | FLOAT | 32-bit float |
| 2 | UINT8 | 8-bit unsigned |
| 3 | INT8 | 8-bit signed |
| 6 | INT32 | 32-bit signed |
| 7 | INT64 | 64-bit signed |
| 10 | FLOAT16 | 16-bit float |
| 11 | DOUBLE | 64-bit float |
| 12 | UINT32 | 32-bit unsigned |
| 13 | UINT64 | 64-bit unsigned |
| 14 | COMPLEX64 | Complex64 |
| 16 | BFLOAT16 | Brain float16 |

## External Tensor Data

Large tensors can be stored in external files:

- Location specified in `external_data` field
- Contains: file path, offset, length
- SHA1 digest optionally available for verification

weight-inspect does NOT load external data - it only notes its existence.

## What weight-inspect Reads

weight-inspect only reads:

1. **Model metadata** - ir_version, producer_name/version, domain
2. **Opset imports** - Required operator versions
3. **Graph structure** - Node count, operation types
4. **Initializers** - Names, shapes, dtypes (not data)
5. **Inputs/Outputs** - Interface shapes

It does NOT read:
- Tensor weight data (raw_data, float_data, etc.)
- External tensor files
- Node attributes (constants)
- Subgraphs

This is intentional for fast, safe structural analysis.

## Structural Hash Contents

For ONNX files, the structural hash includes:

| Field | Included | Notes |
|-------|----------|-------|
| format | ✓ | "onnx" |
| ir_version | ✓ | IR version number |
| opset_imports | ✓ | Operator set versions |
| producer_name | ✓ | Tool name |
| producer_version | ✓ | Tool version |
| domain | ✓ | Model domain |
| node_count | ✓ | Number of operations |
| node_types | ✓ | Operation types (sorted) |
| initializer_names | ✓ | Weight tensor names |
| initializer_shapes | ✓ | Weight tensor shapes |
| initializer_dtypes | ✓ | Weight data types |

## Example: Inspecting an ONNX File

```bash
weight-inspect inspect model.onnx
```

Output:
```
format: ONNX
ir_version: 8
opset_imports: [13]
producer_name: pytorch
producer_version: 2.1
domain: 
tensor_count: 142
metadata_count: 3
structural_hash: abc123...

First 5 tensors:
  1: conv1.weight [64, 1, 7, 7] (FLOAT)
  2: conv1.bias [64] (FLOAT)
  3: bn1.weight [64] (FLOAT)
  ...
```

## Diff Behavior

For ONNX files, `weight-inspect diff` compares:

| Comparison | Description |
|------------|-------------|
| **Initializers/Weights** | Tensor names, shapes, dtypes, byte lengths |
| **Metadata** | IR version, producer info, opset imports |

**Not compared** (graph structure):
- Node operations/types
- Input/output names (only counts compared via metadata)
- Model structure changes

This is intentional - comparing graph structure is complex (what's a meaningful change? opcode differences? structural equivalence?). Weight comparison is sufficient for most use cases.

## References

- [ONNX IR Specification](https://onnx.ai/onnx/docs/IR.html)
- [ONNX Protobuf Definition](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto)
- [ONNX Operator Docs](https://onnx.ai/onnx/docs/Operators.html)
