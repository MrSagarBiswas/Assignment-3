# Parallel Minimum Spanning Tree (MST) Construction using CUDA

**Apr 2025 – May 2025**
Instructor: Prof. Rupesh Nasre — GPU Programming

---

## Project overview

This repository contains a CUDA implementation of a parallel Minimum Spanning Tree (MST) construction based on **Boruvka’s algorithm**. The implementation maps Boruvka’s rounds (component identification, cheapest-edge selection, contraction, and edge filtering) to GPU kernels and focuses on high throughput for dense graph datasets.

Key features:

* Parallel Boruvka-style MST mapped to CUDA kernels.
* GPU-resident **union-find (disjoint-set)** implementation optimized for parallel find/union.
* Weight update, cheapest-edge selection, and contraction performed with data-parallel kernels.
* Data layouts designed for **coalesced global memory access** and minimized branch divergence.
* Optimizations: shared memory usage, warp-level parallelism, kernel fusion, minimized atomics and synchronization to reduce global memory traffic and increase occupancy.
* Correctness validated against a CPU baseline; representative benchmark: **\~12× speedup** vs. sequential CPU on dense graphs.

---

## Repository layout

```
.
├── input/                 # Graph input files (testcases)
├── output/                # Expected MST outputs (same filenames as input)
├── submit/                # Project sources to be compiled and run
│   ├── main.cu            # <-- main CUDA implementation (required)
│   └── compile.sh         # <-- compile script (required)
├── logFile                # Execution & comparison log (created by run.sh)
├── timing_logFile         # Per-test timing log (created by run.sh)
├── run.sh                 # Automates compilation, runs tests in input/, compares with output/, logs timings
└── README.md              # <-- this file
```

> **Important**: Place `main.cu` and `compile.sh` inside `submit/`. `run.sh` expects `submit/compile.sh` to produce a runnable binary (for example `submit/mst_gpu`).

---

## Dependencies

* **NVIDIA GPU + drivers** compatible with the CUDA Toolkit used for building (NVIDIA T4 used for benchmarking).
* **CUDA Toolkit** (nvcc). CUDA 11.x or newer recommended.
* **bash** and standard Unix tools (`diff`, `time`, etc.) used by `run.sh`.
* (Optional) NVIDIA profiling tools: `nsight`, `nvprof`, `nvvp` for performance analysis.

---

## How to compile

Create `submit/compile.sh` (make executable) to compile `submit/main.cu`. Example `compile.sh`:

```bash
#!/usr/bin/env bash
set -e
# adjust -arch / -gencode for your GPU
nvcc -O3 -arch=sm_75 submit/main.cu -o submit/mst_gpu
```

* `sm_75` is appropriate for T4; change to target your GPU (e.g., `sm_61`, `sm_80`, etc.).
* Ensure `compile.sh` is executable: `chmod +x submit/compile.sh`.

---

## How to run (automated)

`run.sh` orchestrates:

1. `submit/compile.sh` to build the binary.
2. Iterates through files in `input/`.
3. Runs the GPU binary on each input to produce an output.
4. Compares produced output against the reference in `output/`.
5. Logs pass/fail details to `logFile` and per-test timings to `timing_logFile`.

Usage:

```bash
chmod +x submit/compile.sh run.sh
./run.sh
```

Outputs:

* `logFile`: per-test execution results (pass/fail, errors, diffs).
* `timing_logFile`: timing and performance numbers per test.

> If your executable produces differently named outputs, update `run.sh` accordingly or adapt the binary to write files matching the `output/` filenames.

---

## Input / output format

* Input graph format: (define in your `main.cu` or README snippet) — typical formats:

  * edge-list: `num_nodes num_edges` followed by `u v weight` lines, OR
  * adjacency format accepted by your binary.
* Output: MST edge list (one edge per line, or as required by `run.sh` comparator).
* Ensure that input filenames in `input/` match expected output filenames in `output/`.

---

## Algorithm & Implementation notes

* **Boruvka rounds**: each round finds cheapest outgoing edge for every component, unions components that are connected, then filters edges to shrink the graph — repeated until MST complete.
* **Union-Find**:

  * Implemented with path compression and union-by-rank/hint adapted for parallelism.
  * Uses atomic operations carefully to avoid excessive contention.
* **Edge filtering & contraction**:

  * Performed in-data-parallel kernels; uses prefix-sum (scan) patterns for compaction.
  * Data layout organizes edges and node metadata to favor coalesced loads/stores.
* **Performance optimizations**:

  * Shared memory buffers for neighbor aggregation and temporary scans.
  * Warp-level intrinsics for fast reductions and ballot/permute operations.
  * Kernel fusion for small consecutive kernels to avoid repeated global memory read/write.
  * Minimize branch divergence by grouping similar-sized work and keeping control flow simple.

---

## Performance & profiling

* Representative speedup: **\~12×** over a sequential CPU baseline on dense graphs (measured on NVIDIA T4).

---

## Testing & correctness

* Validate GPU results against a trusted CPU MST implementation (e.g., sequential Boruvka/Kruskal/Prim).
* `run.sh` automates per-test comparison — ensure comparator tolerates order-insensitive MST outputs (MST edges might be in different orders but still valid). If `run.sh` uses byte-wise `diff`, consider normalizing outputs (sort edges) before comparison.

---
