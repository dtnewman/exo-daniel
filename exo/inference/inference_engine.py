import numpy as np
import os
from exo.helpers import DEBUG  # Make sure to import DEBUG

from typing import Tuple, Optional
from abc import ABC, abstractmethod
from .shard import Shard
class InferenceEngine(ABC):
  @abstractmethod
  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    pass
  
  @abstractmethod
  async def sample(self, x: np.ndarray) -> np.ndarray:
    pass

  @abstractmethod
  async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
    pass

  @abstractmethod
  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> np.ndarray:
    pass
  
  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, inference_state: Optional[str] = None) -> np.ndarray:
    tokens = await self.encode(shard, prompt)
    output_data = await self.infer_tensor(request_id, shard, tokens, inference_state)
    return output_data 


def inference_engine_classes():
  from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
  from exo.inference.mlx.sharded_inference_engine import MLXDynamicShardInferenceEngine
  from exo.inference.dummy_inference_engine import DummyInferenceEngine
  return {
    "mlx": MLXDynamicShardInferenceEngine,
    "tinygrad": TinygradDynamicShardInferenceEngine,
    "dummy": DummyInferenceEngine,
  }

def get_inference_engine(inference_engine_name: str, shard_downloader: 'ShardDownloader'):
  if DEBUG >= 2:
    print(f"get_inference_engine called with: {inference_engine_name}")
  if inference_engine_name == "tinygrad":
    import tinygrad.helpers
    tinygrad.helpers.DEBUG.value = int(os.getenv("TINYGRAD_DEBUG", default="0"))
  classes = inference_engine_classes()
  if inference_engine_name in classes:
    return classes[inference_engine_name](shard_downloader)
  raise ValueError(f"Unsupported inference engine: {inference_engine_name}")
