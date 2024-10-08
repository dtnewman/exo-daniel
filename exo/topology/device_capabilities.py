from exo import DEBUG
from dataclasses import dataclass, asdict
import subprocess
import psutil

TFLOPS = 1.00


@dataclass
class DeviceFlops:
  # units of TFLOPS
  fp32: float
  fp16: float
  int8: float

  def __str__(self):
    return f"fp32: {self.fp32 / TFLOPS:.2f} TFLOPS, fp16: {self.fp16 / TFLOPS:.2f} TFLOPS, int8: {self.int8 / TFLOPS:.2f} TFLOPS"

  def to_dict(self):
    return asdict(self)


@dataclass
class DeviceCapabilities:
  model: str
  chip: str
  memory: int
  flops: DeviceFlops

  def __str__(self):
    return f"Model: {self.model}. Chip: {self.chip}. Memory: {self.memory}MB. Flops: {self.flops}"

  def __post_init__(self):
    if isinstance(self.flops, dict):
      self.flops = DeviceFlops(**self.flops)

  def to_dict(self):
    return {"model": self.model, "chip": self.chip, "memory": self.memory, "flops": self.flops.to_dict()}


UNKNOWN_DEVICE_CAPABILITIES = DeviceCapabilities(model="Unknown Model", chip="Unknown Chip", memory=0, flops=DeviceFlops(fp32=0, fp16=0, int8=0))

CHIP_FLOPS = {
  # Source: https://www.cpu-monkey.com
  # Note: currently no distinction between variants of M3 Max and M3 Pro, we pick the lower one to be conservative
  ### M chips
  "apple m1": DeviceFlops(fp32=2.29*TFLOPS, fp16=4.58*TFLOPS, int8=9.16*TFLOPS),
  "apple m1 pro": DeviceFlops(fp32=5.30*TFLOPS, fp16=10.60*TFLOPS, int8=21.20*TFLOPS),
  "apple m1 max": DeviceFlops(fp32=10.60*TFLOPS, fp16=21.20*TFLOPS, int8=42.40*TFLOPS),
  "apple m1 ultra": DeviceFlops(fp32=21.20*TFLOPS, fp16=42.40*TFLOPS, int8=84.80*TFLOPS),
  "apple m2": DeviceFlops(fp32=3.55*TFLOPS, fp16=7.10*TFLOPS, int8=14.20*TFLOPS),
  "apple m2 pro": DeviceFlops(fp32=5.68*TFLOPS, fp16=11.36*TFLOPS, int8=22.72*TFLOPS),
  "apple m2 max": DeviceFlops(fp32=13.49*TFLOPS, fp16=26.98*TFLOPS, int8=53.96*TFLOPS),
  "apple m2 ultra": DeviceFlops(fp32=26.98*TFLOPS, fp16=53.96*TFLOPS, int8=107.92*TFLOPS),
  "apple m3": DeviceFlops(fp32=3.55*TFLOPS, fp16=7.10*TFLOPS, int8=14.20*TFLOPS),
  "apple m3 max": DeviceFlops(fp32=14.20*TFLOPS, fp16=28.40*TFLOPS, int8=56.80*TFLOPS),
  "apple m3 pro": DeviceFlops(fp32=4.97*TFLOPS, fp16=9.94*TFLOPS, int8=19.88*TFLOPS),
  "apple m4": DeviceFlops(fp32=3.55*TFLOPS, fp16=7.10*TFLOPS, int8=14.20*TFLOPS),
  ### A chips
  "apple a13": DeviceFlops(fp32=0.69*TFLOPS, fp16=1.38*TFLOPS, int8=2.76*TFLOPS),
  "apple a14": DeviceFlops(fp32=0.75*TFLOPS, fp16=1.50*TFLOPS, int8=3.00*TFLOPS),
  "apple a15": DeviceFlops(fp32=1.37*TFLOPS, fp16=2.74*TFLOPS, int8=5.48*TFLOPS),
  "apple a16": DeviceFlops(fp32=1.79*TFLOPS, fp16=3.58*TFLOPS, int8=7.16*TFLOPS),
  "apple a17 pro": DeviceFlops(fp32=2.15*TFLOPS, fp16=4.30*TFLOPS, int8=8.60*TFLOPS),
  ### NVIDIA GPUs
  # RTX 40 series
  "nvidia geforce rtx 4090": DeviceFlops(fp32=82.58*TFLOPS, fp16=165.16*TFLOPS, int8=330.32*TFLOPS),
  "nvidia geforce rtx 4080": DeviceFlops(fp32=48.74*TFLOPS, fp16=97.48*TFLOPS, int8=194.96*TFLOPS),
  "nvidia geforce rtx 4080 super": DeviceFlops(fp32=52.0*TFLOPS, fp16=104.0*TFLOPS, int8=208.0*TFLOPS),
  "nvidia geforce rtx 4070 ti super": DeviceFlops(fp32=40.0*TFLOPS, fp16=80.0*TFLOPS, int8=160.0*TFLOPS),
  "nvidia geforce rtx 4070 ti": DeviceFlops(fp32=39.43*TFLOPS, fp16=78.86*TFLOPS, int8=157.72*TFLOPS),
  "nvidia geforce rtx 4070 super": DeviceFlops(fp32=30.0*TFLOPS, fp16=60.0*TFLOPS, int8=120.0*TFLOPS),
  "nvidia geforce rtx 4070": DeviceFlops(fp32=29.0*TFLOPS, fp16=58.0*TFLOPS, int8=116.0*TFLOPS),
  "nvidia geforce rtx 4060 ti 16gb": DeviceFlops(fp32=22.0*TFLOPS, fp16=44.0*TFLOPS, int8=88.0*TFLOPS),
  # RTX 30 series
  "nvidia geforce rtx 3050": DeviceFlops(fp32=9.11*TFLOPS, fp16=18.22*TFLOPS, int8=36.44*TFLOPS),
  "nvidia geforce rtx 3060": DeviceFlops(fp32=13.0*TFLOPS, fp16=26.0*TFLOPS, int8=52.0*TFLOPS),
  "nvidia geforce rtx 3060 ti": DeviceFlops(fp32=16.2*TFLOPS, fp16=32.4*TFLOPS, int8=64.8*TFLOPS),
  "nvidia geforce rtx 3070": DeviceFlops(fp32=20.3*TFLOPS, fp16=40.6*TFLOPS, int8=81.2*TFLOPS),
  "nvidia geforce rtx 3070 ti": DeviceFlops(fp32=21.8*TFLOPS, fp16=43.6*TFLOPS, int8=87.2*TFLOPS),
  "nvidia geforce rtx 3080 (10 gb)": DeviceFlops(fp32=29.8*TFLOPS, fp16=59.6*TFLOPS, int8=119.2*TFLOPS),
  "nvidia geforce rtx 3080 (12 gb)": DeviceFlops(fp32=30.6*TFLOPS, fp16=61.2*TFLOPS, int8=122.4*TFLOPS),
  "nvidia geforce rtx 3080 ti": DeviceFlops(fp32=34.1*TFLOPS, fp16=68.2*TFLOPS, int8=136.4*TFLOPS),
  "nvidia geforce rtx 3090": DeviceFlops(fp32=35.6*TFLOPS, fp16=71.2*TFLOPS, int8=142.4*TFLOPS),
  "nvidia geforce rtx 3090 ti": DeviceFlops(fp32=40.0*TFLOPS, fp16=80.0*TFLOPS, int8=160.0*TFLOPS),
  # RTX 20 series
  "nvidia geforce rtx 2060": DeviceFlops(fp32=6.45*TFLOPS, fp16=12.9*TFLOPS, int8=25.8*TFLOPS),
  "nvidia geforce rtx 2060 super": DeviceFlops(fp32=7.2*TFLOPS, fp16=14.4*TFLOPS, int8=28.8*TFLOPS),
  "nvidia geforce rtx 2070": DeviceFlops(fp32=7.46*TFLOPS, fp16=14.93*TFLOPS, int8=29.86*TFLOPS),
  "nvidia geforce rtx 2070 super": DeviceFlops(fp32=9.06*TFLOPS, fp16=18.12*TFLOPS, int8=36.24*TFLOPS),
  "nvidia geforce rtx 2080": DeviceFlops(fp32=10.07*TFLOPS, fp16=20.14*TFLOPS, int8=40.28*TFLOPS),
  "nvidia geforce rtx 2080 super": DeviceFlops(fp32=11.15*TFLOPS, fp16=22.30*TFLOPS, int8=44.60*TFLOPS),
  "nvidia titan rtx": DeviceFlops(fp32=16.31*TFLOPS, fp16=32.62*TFLOPS, int8=65.24*TFLOPS),
  # QUATRO RTX Ampere series
  "nvidia quatro rtx a2000": DeviceFlops(fp32=7.99*TFLOPS, fp16=7.99*TFLOPS, int8=31.91*TFLOPS),
  "nvidia quatro rtx a4000": DeviceFlops(fp32=19.17*TFLOPS, fp16=19.17*TFLOPS, int8=76.68*TFLOPS),
  "nvidia quatro rtx a4500": DeviceFlops(fp32=23.65*TFLOPS, fp16=23.65*TFLOPS, int8=94.6*TFLOPS),
  "nvidia quatro rtx a5000": DeviceFlops(fp32=27.8*TFLOPS, fp16=27.8*TFLOPS, int8=111.2*TFLOPS),
  "nvidia quatro rtx a6000": DeviceFlops(fp32=38.71*TFLOPS, fp16=38.71*TFLOPS, int8=154.84*TFLOPS),
  # Common Server GPUs
  "nvidia a40 48gb pcie": DeviceFlops(fp32=37.4*TFLOPS, fp16=149.7*TFLOPS, int8=299.3*TFLOPS),
  "nvidia a100 40gb pcie": DeviceFlops(fp32=19.5*TFLOPS, fp16=312.0*TFLOPS, int8=624.0*TFLOPS),
  "nvidia a800 40gb pcie": DeviceFlops(fp32=19.5*TFLOPS, fp16=312.0*TFLOPS, int8=624.0*TFLOPS),
  "nvidia a100 80gb pcie": DeviceFlops(fp32=19.5*TFLOPS, fp16=312.0*TFLOPS, int8=624.0*TFLOPS),
  "nvidia a800 80gb pcie": DeviceFlops(fp32=19.5*TFLOPS, fp16=312.0*TFLOPS, int8=624.0*TFLOPS),
  "nvidia a100 80gb sxm": DeviceFlops(fp32=19.5*TFLOPS, fp16=312.0*TFLOPS, int8=624.0*TFLOPS),
  "nvidia a800 80gb sxm": DeviceFlops(fp32=19.5*TFLOPS, fp16=312.0*TFLOPS, int8=624.0*TFLOPS),
  "nvidia t1000 8gb": DeviceFlops(fp32=2.5 * TFLOPS, fp16=5.0 * TFLOPS, int8=10.0 * TFLOPS),
  "nvidia quadro p6000": DeviceFlops(fp32=12.5 * TFLOPS, fp16=25.0 * TFLOPS, int8=100.0 * TFLOPS),
  "nvidia quadro m4000": DeviceFlops(fp32=2.5 * TFLOPS, fp16=5.0 * TFLOPS, int8=10.0 * TFLOPS),
  "nvidia quadro m2000": DeviceFlops(fp32=0.5 * TFLOPS, fp16=1.0 * TFLOPS, int8=2.0 * TFLOPS),
  "nvidia quadro p400": DeviceFlops(fp32=0.641 * TFLOPS, fp16=1.282 * TFLOPS, int8=2.564 * TFLOPS),
  "nvidia a10": DeviceFlops(fp32=31.2 * TFLOPS, fp16=62.5 * TFLOPS, int8=2.5 * TFLOPS),
  # ... add more devices if needed ...
  ### AMD GPUs
  # RX 6000 series
  "amd radeon rx 6900 xt": DeviceFlops(fp32=23.04*TFLOPS, fp16=46.08*TFLOPS, int8=92.16*TFLOPS),
  "amd radeon rx 6800 xt": DeviceFlops(fp32=20.74*TFLOPS, fp16=41.48*TFLOPS, int8=82.96*TFLOPS),
  "amd radeon rx 6800": DeviceFlops(fp32=16.17*TFLOPS, fp16=32.34*TFLOPS, int8=64.68*TFLOPS),
  "amd radeon rx 6700 xt": DeviceFlops(fp32=13.21*TFLOPS, fp16=26.42*TFLOPS, int8=52.84*TFLOPS),
  "amd radeon rx 6700": DeviceFlops(fp32=11.4*TFLOPS, fp16=22.8*TFLOPS, int8=45.6*TFLOPS),
  "amd radeon rx 6600 xt": DeviceFlops(fp32=10.6*TFLOPS, fp16=21.2*TFLOPS, int8=42.4*TFLOPS),
  "amd radeon rx 6600": DeviceFlops(fp32=8.93*TFLOPS, fp16=17.86*TFLOPS, int8=35.72*TFLOPS),
  "amd radeon rx 6500 xt": DeviceFlops(fp32=5.77*TFLOPS, fp16=11.54*TFLOPS, int8=23.08*TFLOPS),
  "amd radeon rx 6400": DeviceFlops(fp32=3.57*TFLOPS, fp16=7.14*TFLOPS, int8=14.28*TFLOPS),
  # RX 7000 series
  "amd radeon rx 7900 xtx": DeviceFlops(fp32=61.4*TFLOPS, fp16=122.8*TFLOPS, int8=245.6*TFLOPS),
  "amd radeon rx 7900 xt": DeviceFlops(fp32=53.4*TFLOPS, fp16=106.8*TFLOPS, int8=213.6*TFLOPS),
  "amd radeon rx 7800 xt": DeviceFlops(fp32=42.6*TFLOPS, fp16=85.2*TFLOPS, int8=170.4*TFLOPS),
  "amd radeon rx 7700 xt": DeviceFlops(fp32=34.2*TFLOPS, fp16=68.4*TFLOPS, int8=136.8*TFLOPS),
  "amd radeon rx 7600": DeviceFlops(fp32=21.5*TFLOPS, fp16=43.0*TFLOPS, int8=86.0*TFLOPS),
  "amd radeon rx 7500": DeviceFlops(fp32=16.2*TFLOPS, fp16=32.4*TFLOPS, int8=64.8*TFLOPS),
  ### Qualcomm embedded chips: TODO
}
CHIP_FLOPS.update({f"laptop gpu {key}": value for key, value in CHIP_FLOPS.items()})
CHIP_FLOPS.update({f"{key} laptop gpu": value for key, value in CHIP_FLOPS.items()})


def device_capabilities() -> DeviceCapabilities:
  if psutil.MACOS:
    return mac_device_capabilities()
  elif psutil.LINUX:
    return linux_device_capabilities()
  else:
    return DeviceCapabilities(
      model="Unknown Device",
      chip="Unknown Chip",
      memory=psutil.virtual_memory().total // 2**20,
      flops=DeviceFlops(fp32=0, fp16=0, int8=0),
    )


def mac_device_capabilities() -> DeviceCapabilities:
  # Fetch the model of the Mac using system_profiler
  model = subprocess.check_output(["system_profiler", "SPHardwareDataType"]).decode("utf-8")
  model_line = next((line for line in model.split("\n") if "Model Name" in line), None)
  model_id = model_line.split(": ")[1] if model_line else "Unknown Model"
  chip_line = next((line for line in model.split("\n") if "Chip" in line), None)
  chip_id = chip_line.split(": ")[1] if chip_line else "Unknown Chip"
  memory_line = next((line for line in model.split("\n") if "Memory" in line), None)
  memory_str = memory_line.split(": ")[1] if memory_line else "Unknown Memory"
  memory_units = memory_str.split()
  memory_value = int(memory_units[0])
  if memory_units[1] == "GB":
    memory = memory_value*1024
  else:
    memory = memory_value

  # Assuming static values for other attributes for demonstration
  foo = "BAR"
  return DeviceCapabilities(model=model_id, chip=chip_id, memory=memory, flops=CHIP_FLOPS.get(chip_id.lower(), DeviceFlops(fp32=0, fp16=0, int8=0)))


def linux_device_capabilities() -> DeviceCapabilities:
  import psutil
  from tinygrad import Device

  if DEBUG >= 2: print(f"tinygrad {Device.DEFAULT=}")
  if Device.DEFAULT == "CUDA" or Device.DEFAULT == "NV" or Device.DEFAULT == "GPU":
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(handle).upper()
    gpu_memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

    if DEBUG >= 2: print(f"NVIDIA device {gpu_name=} {gpu_memory_info=}")

    return DeviceCapabilities(
      model=f"Linux Box ({gpu_name})",
      chip=gpu_name,
      memory=gpu_memory_info.total // 2**20,
      flops=CHIP_FLOPS.get(gpu_name.lower(), DeviceFlops(fp32=0, fp16=0, int8=0)),
    )
  elif Device.DEFAULT == "AMD":
    # TODO AMD support
    return DeviceCapabilities(
      model="Linux Box (AMD)",
      chip="Unknown AMD",
      memory=psutil.virtual_memory().total // 2**20,
      flops=DeviceFlops(fp32=0, fp16=0, int8=0),
    )
  else:
    return DeviceCapabilities(
      model=f"Linux Box (Device: {Device.DEFAULT})",
      chip=f"Unknown Chip (Device: {Device.DEFAULT})",
      memory=psutil.virtual_memory().total // 2**20,
      flops=DeviceFlops(fp32=0, fp16=0, int8=0),
    )
