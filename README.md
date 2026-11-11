# Deep Dive: Systems-Level GPU Performance Engineering – Parallel Mandelbrot Rendering via Custom CUDA Kernels

## 1. Motivation & Problem Statement
While high-level deep learning frameworks abstract away hardware complexities, true performance engineering requires stepping down to the bare metal. Consumer-grade hardware—such as an **NVIDIA RTX 3050 with 6 GB VRAM**—provides a constrained yet powerful playground to explore what happens when software meets raw hardware architecture.

This project moves away from neural networks to focus on **pure computational throughput and parallel architecture**. By rendering the Mandelbrot set across escalating resolutions (up to 4K Ultra HD), this benchmark investigates how massively parallel GPU execution scales compared to vectorized CPU processing, and how low-level thread management unlocks performance orders of magnitude beyond naive implementations.

---

## 2. Theoretical Background: The Anatomy of "Embarrassingly Parallel" Compute
The Mandelbrot set is defined by the iterative complex quadratic polynomial:

zₙ₊₁ = zₙ² + c

To render an image, every single pixel must independently evaluate whether its corresponding complex coordinate c escapes toward infinity or remains bounded within a maximum iteration limit. 

* **Why it matters:** This is the textbook definition of an *embarrassingly parallel* workload. There is zero inter-thread communication or data dependency between pixels. 
* **The Hardware Challenge:** On a CPU, pixels must be processed sequentially or via vector registers (SIMD), creating a massive bottleneck for high-resolution grids. On a GPU, thousands of CUDA cores can evaluate millions of pixels simultaneously—provided the execution grid, thread blocks, and memory access patterns are engineered correctly.

---

## 3. Test Environment
* **CPU:** Intel Core i3-12100F (4 cores / 8 threads) — 16 GB System RAM
* **GPU:** NVIDIA GeForce RTX 3050 — 6 GB VRAM
* **Software:** Ubuntu 22.04 LTS, Python, Numba (`numba.cuda`), NumPy, PyTorch, Matplotlib, Pynvml

---

## 4. The Experimental Journey & Practical Observations ("What I Experienced")

During the resolution-scaling stress tests, several distinct performance dynamics became apparent:

### Phase 1: The CPU Vectorization Wall
At lower resolutions (HD 1280x720), NumPy's vectorized CPU operations handle the workload reasonably well (~0.7 seconds). However, as pixel counts scale up quadratically toward 4K, the sequential/vectorized CPU approach bogs down significantly, climbing to **7.91 seconds** at Ultra HD.

### Phase 2: Unleashing Custom CUDA Kernels (`@cuda.jit`)
By writing a custom low-level CUDA kernel where every thread maps directly to an individual pixel coordinate, the execution paradigm completely shifts. 
* **What I Observed:** The GPU bypasses framework overhead entirely. At HD resolution, the CUDA kernel renders the fractal in a mere **0.0115 seconds** (61x faster than the CPU).
* **The Scaling Effect:** As the workload increases to 4K Ultra HD (8.29 million pixels), the GPU's thousands of parallel cores absorb the compute pressure effortlessly, completing the render in just **0.0741 seconds** and achieving an astounding **106.8x speedup**.

### Phase 3: Real-Time Hardware Profiling & Throughput
Using live background monitoring (`psutil`, `pynvml` and `nvtop`), the system captured real-time metrics during execution. The GPU core utilization spiked immediately to 100% during kernel execution, while maintaining a stable thermal and VRAM footprint, highlighting the raw efficiency of hardware-accelerated grid processing.

<img width="4800" height="2700" alt="mandelbrot_4k_cuda_1" src="https://github.com/user-attachments/assets/5c12202c-fd98-41e8-9a2e-b185bd96f640" />

---
## 5. Comprehensive Benchmark Results

| Resolution | Total Pixels | CPU Time (NumPy) | CUDA Time (GPU) | Speedup Factor | Throughput (GPU) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HD** (1280x720) | ~0.92 MP | 0.6989s | 0.0115s | **61.0x** | ~80 MP/s |
| **Full HD** (1920x1080) | ~2.07 MP | 1.7257s | 0.0221s | **78.2x** | ~93 MP/s |
| **QHD / 2K** (2560x1440) | ~3.69 MP | 3.2069s | 0.0391s | **82.0x** | ~94 MP/s |
| **Ultra HD / 4K** (3840x2160) | ~8.29 MP | 7.9125s | 0.0741s | **106.8x** | ~112 MP/s |

<img width="3000" height="1800" alt="resolution_scaling_benchmark" src="https://github.com/user-attachments/assets/0c7dd4bf-e25e-4dbe-be28-ea3615f4b947" />

---

## 6. Real-World Implications: Why Low-Level Parallelism Matters

While rendering fractals is visually striking, the underlying principles apply directly to high-performance computing (HPC), real-time graphics pipelines, simulation physics, and edge AI inference:
* **Throughput Scaling:** Hardware bottlenecks are rarely solved by raw clock speed alone; they require matching the algorithm's data layout to the hardware's parallelism (Grid-Stride loops, thread blocks, memory coalescing).
* **Systems Engineering Mindset:** Moving from high-level abstractions to writing custom CUDA kernels bridges the gap between theoretical mathematics and practical production engineering.

---

## 7. Conclusion
This project demonstrates that unlocking the full potential of consumer hardware requires understanding how software instructions translate into hardware execution. By bypassing framework abstractions and implementing custom CUDA kernels, even an entry-level RTX 3050 achieves over **106.8x performance gains** over optimized CPU code—proving that systems-level engineering is the ultimate differentiator in performance-critical applications.

---

### Future Work
* Implementing an interactive, real-time deep-zoom navigation loop using CUDA texture memory.
* Writing a native C++ CUDA extension (`cppimport` / Pybind11) for deeper systems-level integration.
