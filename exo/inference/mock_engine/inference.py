import json
import time
from exo.inference.shard import Shard
from exo.inference.inference_engine import InferenceEngine
from typing import List, Optional, Tuple
import numpy as np
import time
import datetime

class MockInferenceEngine(InferenceEngine):
    """
    Mock inference engine for testing and development.
    """
    
    def __init__(self, input_data: List[int] = None, response: List[int] = None, latency: float = 1):
        # default_response = never gonna let you down
        default_response = [2646, 16926, 1095, 499, 1523, 128009]
        # default_input_data = finish this sentence: never gonna give you up
        default_input_data = [51585, 65267, 53498, 52743, 63954, 61797, 56691, 61758, 60298, 62148, 59485, 57423, 61918, 57683, 49417, 61158, 62351, 52365, 52251]
        
        self.input_data = input_data or default_input_data
        self.response = iter(response or default_response)
        self.original_response = response or default_response
        self.latency = latency
    
    def reset_response(self):
        self.response = iter(self.original_response)

    async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        array = self.input_data
        time.sleep(self.latency)
        self.reset_response()
        return np.array(array), json.dumps({"start_pos": 0, "n_captured_toks": len(array)}), False

    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        time.sleep(self.latency)
        next_token = next(self.response, None)

        cached_iids = {"input_ids": self.input_data + [next_token]}
        is_finished = next_token == 128009
        if is_finished:
            self.reset_response()
        response = np.array([next_token]), json.dumps({"cached_iids": cached_iids}), is_finished
        print(f"\nresponse: {response}")
        return response

# Dynamically modify the __name__ attribute of the class so that it acts like the pytorch inference engine.
# This is necessary so that it can pick up settings related to the pytorch inference engine (e.g., eligible models).
MockInferenceEngine.__name__ = "PyTorchDynamicShardInferenceEngine"
