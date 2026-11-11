import time
import matplotlib.pyplot as plt
from numba import cuda
import numpy as np
import pynvml
import psutil
import torch
import threading


# 1. HARDWARE-MONITOR (RAM, VRAM, Auslastung, Temp)

class AdvancedHardwareMonitor:

  def __init__(self, interval=0.005):
    self.interval = interval
    self.stop_event = threading.Event()
    self.ram_history = []
    self.vram_history = []
    self.util_history = []
    self.temp_history = []
    self.thread = None

    try:
      pynvml.nvmlInit()
      self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
      self.nvml_active = True
    except Exception:
      self.nvml_active = False

  def _monitor_loop(self):
    process = psutil.Process()
    while not self.stop_event.is_set():
      ram_mb = process.memory_info().rss / (1024 ** 2)
      self.ram_history.append(ram_mb)

      if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        self.vram_history.append(vram_mb)
      else:
        self.vram_history.append(0)

      if self.nvml_active:
        try:
          util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
          self.util_history.append(util.gpu)
          temp = pynvml.nvmlDeviceGetTemperature(
              self.handle, pynvml.NVML_TEMPERATURE_GPU
          )
          self.temp_history.append(temp)
        except Exception:
          pass

      time.sleep(self.interval)

  def start(self):
    self.ram_history.clear()
    self.vram_history.clear()
    self.util_history.clear()
    self.temp_history.clear()
    self.stop_event.clear()
    self.thread = threading.Thread(target=self._monitor_loop)
    self.thread.start()

  def stop(self):
    self.stop_event.set()
    if self.thread:
      self.thread.join()
    if self.nvml_active:
      try:
        pynvml.nvmlShutdown()
      except Exception:
        pass


# 2. IMPLEMENTIERUNGEN (CPU NumPy & CUDA Kernel)

def mandelbrot_cpu(width, height, max_iter, xmin, xmax, ymin, ymax):
  """Vektorisierte CPU-Berechnung mit NumPy."""
  x = np.linspace(xmin, xmax, width)
  y = np.linspace(ymin, ymax, height)
  c = x + y[:, None] * 1j
  z = np.zeros_like(c)
  output = np.zeros(c.shape, dtype=np.int32)

  for i in range(max_iter):
    mask = np.abs(z) <= 2.0
    if not np.any(mask):
      break
    z[mask] = z[mask] ** 2 + c[mask]
    output[mask] = i

  return output


@cuda.jit(device=True)
def mandelbrot_pixel(c_real, c_imag, max_iter):
  z_real = 0.0
  z_imag = 0.0
  for i in range(max_iter):
    if z_real * z_real + z_imag * z_imag > 4.0:
      return i
    xtemp = z_real * z_real - z_imag * z_imag + c_real
    z_imag = 2.0 * z_real * z_imag + c_imag
    z_real = xtemp
  return max_iter


@cuda.jit
def mandelbrot_cuda_kernel(output, max_iter, xmin, xmax, ymin, ymax):
  startX, startY = cuda.grid(2)
  gridWidth, gridHeight = cuda.gridsize(2)

  pixelWidth = (xmax - xmin) / output.shape[1]
  pixelHeight = (ymax - ymin) / output.shape[0]

  for x in range(startX, output.shape[1], gridWidth):
    for y in range(startY, output.shape[0], gridHeight):
      c_real = xmin + x * pixelWidth
      c_imag = ymin + y * pixelHeight
      output[y, x] = mandelbrot_pixel(c_real, c_imag, max_iter)



# 3 HAUPTPROGRAMM & DYNAMISCHER STRESSTEST

def main():
  if not cuda.is_available():
    print("❌ Keine CUDA-GPU gefunden!")
    return

  # Stufenweise Erhöhung der Auflösung, um den Leistungsunterschied zu demonstrieren
  resolutions = [
      ("HD", 1280, 720),
      ("Full HD", 1920, 1080),
      ("QHD / 2K", 2560, 1440),
      ("Ultra HD / 4K", 3840, 2160),
  ]

  max_iter = 200
  xmin, xmax, ymin, ymax = -2.0, 0.5, -1.25, 1.25

  results = []

  print(f"{'Auflösung':<15} | {'CPU Zeit (s)':<15} | {'CUDA Zeit (s)':<15} | {'Speedup':<10}")
  print("-" * 65)

  for name, width, height in resolutions:
    total_pixels = width * height
    total_megapixels = total_pixels / 1e6

    # CPU Test
    # Bei 4K auf der CPU ggf. etwas Geduld, da NumPy hier sequenziell/vektoriell arbeitet
    start_cpu = time.time()
    _ = mandelbrot_cpu(width, height, max_iter, xmin, xmax, ymin, ymax)
    cpu_time = time.time() - start_cpu

    # CUDA GPU Test
    host_output = np.zeros((height, width), dtype=np.int32)
    d_output = cuda.to_device(host_output)

    threadsperblock = (16, 16)
    blockspergrid = (
        (width + threadsperblock[0] - 1) // threadsperblock[0],
        (height + threadsperblock[1] - 1) // threadsperblock[1],
    )

    # JIT Warmup
    mandelbrot_cuda_kernel[blockspergrid, threadsperblock](
        d_output, max_iter, xmin, xmax, ymin, ymax
    )
    cuda.synchronize()

    start_gpu = time.time()
    mandelbrot_cuda_kernel[blockspergrid, threadsperblock](
        d_output, max_iter, xmin, xmax, ymin, ymax
    )
    cuda.synchronize()
    gpu_time = time.time() - start_gpu

    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    results.append((name, total_megapixels, cpu_time, gpu_time, speedup))

    print(
        f"{name:<15} | {cpu_time:>13.4f}s | {gpu_time:>13.4f}s |"
        f" {speedup:>8.1f}x"
    )

# PLOT DER SKALIERUNG (CPU vs GPU über Auflösungen) ---
  names = [r[0] for r in results]
  cpu_times = [r[2] for r in results]
  gpu_times = [r[3] for r in results]

  plt.figure(figsize=(10, 6), dpi=300)
  plt.plot(names, cpu_times, marker="o", color="crimson", label="CPU (NumPy)", linewidth=2)
  plt.plot(names, gpu_times, marker="s", color="dodgerblue", label="CUDA GPU (RTX 3050)", linewidth=2)

# Logarithmische Skala, da der CPU-Abfall massiv ist
  plt.yscale("log") 
  plt.title("Skalierungs-Stresstest: CPU vs. CUDA GPU bei steigender Auflösung", fontsize=12, fontweight="bold")
  plt.xlabel("Bildauflösung", fontsize=10)
  plt.ylabel("Rechenzeit in Sekunden (Log-Skala)", fontsize=10)
  plt.grid(True, linestyle=":", alpha=0.6)
  plt.legend(fontsize=10)
  plt.tight_layout()

  plt.savefig("resolution_scaling_benchmark.png", dpi=300)
  print("\n✅ Skalierungs-Benchmark-Plot erfolgreich gespeichert unter: resolution_scaling_benchmark.png")


if __name__ == "__main__":
  main()