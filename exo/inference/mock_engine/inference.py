import json
from exo.inference.shard import Shard
from exo.inference.inference_engine import InferenceEngine
from typing import Optional, Tuple
import numpy as np


class MockInferenceEngine(InferenceEngine):
    
    def __init__(self):
        self.response = iter([2646, 16926, 1095, 499, 1523, 128009])
        self.input_data = [51585, 65267, 53498, 52743, 63954, 61797, 56691, 61758, 60298, 62148, 59485, 57423, 61918, 57683, 49417, 61158, 62351, 52365, 52251]

    async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        array = self.input_data
        return np.array(array), json.dumps({"start_pos": 0, "n_captured_toks": len(array)}), False

    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        next_token = self.response.__next__()
        cached_iids = {"input_ids": self.input_data + [next_token]}
        is_finished = next_token == 128009
        response = np.array([next_token]), json.dumps({"cached_iids": cached_iids}), is_finished
        return response

# Dynamically modify the __name__ attribute of the class so that it acts like the pytorch inference engine
MockInferenceEngine.__name__ = "PyTorchDynamicShardInferenceEngine"
