import numpy as np
import os

from typing import Tuple, Optional
from abc import ABC, abstractmethod
from .shard import Shard


class InferenceEngine(ABC):
  @abstractmethod
  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
    pass

  @abstractmethod
  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
    pass


def get_inference_engine(inference_engine_name: str, shard_downloader: 'ShardDownloader', mock_inference_engine_sleep_time: float = 1, mock_inference_engine_throughput: int = 1):
  if inference_engine_name == "mlx":
    from exo.inference.mlx.sharded_inference_engine import MLXDynamicShardInferenceEngine

    return MLXDynamicShardInferenceEngine(shard_downloader)
  elif inference_engine_name == "tinygrad":
    from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
    import tinygrad.helpers
    tinygrad.helpers.DEBUG.value = int(os.getenv("TINYGRAD_DEBUG", default="0"))

    return TinygradDynamicShardInferenceEngine(shard_downloader)
  elif inference_engine_name == "pytorch":
    from exo.inference.pytorch.inference import PyTorchDynamicShardInferenceEngine
    return PyTorchDynamicShardInferenceEngine(shard_downloader)
  elif inference_engine_name == "mock":
    from exo.inference.mock_engine.inference import MockInferenceEngine

    return MockInferenceEngine(throughput=mock_inference_engine_throughput, sleep_time=mock_inference_engine_sleep_time)
  else:
    raise ValueError(f"Inference engine {inference_engine_name} not supported")
