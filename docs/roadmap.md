# Roadmap

## Summary

The repository already has a usable compiler core. The roadmap below focuses on
the work needed to turn that core into a stronger end-to-end FPGA deployment
story.

## Current Baseline

Today the repo already supports:

- IR graph construction and operator registration
- Primitive operator validation and coarse FPGA cost heuristics
- ONNX ingestion with PyTorch and TensorFlow wrappers
- Quantization spec attachment
- Representative-dataset calibration utilities
- Operator-level HLS template rendering
- Device presets and validation

That is a meaningful foundation, and it gives the roadmap a concrete starting
point instead of a blank slate.

## Near-Term Efforts

### 1. High-Level Model Lowering

Move from parser coverage toward stronger lowering of sequence-model constructs:

- Expand recurrent-model handling
- Lower high-level layers into primitive operator graphs consistently
- Make graph outputs easier to validate across frameworks

### 2. Graph Optimization

Add compiler passes that make the IR more hardware-aware:

- Operator fusion where it improves throughput or memory reuse
- Scheduling hints and latency-aware transforms
- Buffer planning and state reuse for sequence workloads

### 3. Quantization And Calibration Integration

Build on the new calibration utilities by connecting them more tightly to
deployment-oriented quantization:

- Persist calibration summaries
- Add tensor-wise or layer-wise observer flows
- Feed calibration outputs directly into graph value quantization metadata

## Medium-Term Efforts

### 4. Codegen Expansion

Move from isolated operator templates toward richer backend generation:

- Graph-level HLS emission
- Template composition across multiple operators
- Toolchain wrappers for simulation and synthesis

### 5. Hardware Validation

Strengthen the link between compiler output and real FPGA execution:

- Board-specific benchmarking
- Validation against reference framework outputs
- Latency and resource reporting against device presets

### 6. Developer Experience

Make the project easier to adopt and extend:

- Better packaging metadata
- CLI workflows for parse, validate, calibrate, and render
- Tutorials and reproducible examples

## Longer-Term Direction

The broader direction is still the same: an open, research-friendly compiler
path for sequence-model acceleration on FPGA hardware. The difference now is
that the repo already contains the beginnings of that stack, so future work can
focus on layering capability instead of replacing the foundation.
