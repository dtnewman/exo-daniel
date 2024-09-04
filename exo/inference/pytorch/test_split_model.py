import torch
import torch.nn as nn
import asyncio
import gc
from transformers import AutoModelForCausalLM, AutoConfig, Qwen2ForCausalLM
from exo.api.chatgpt_api import resolve_tokenizer
from typing import Tuple, Optional
import re
from exo.inference.pytorch.utils import sample_logits, top_k_sampling

TEMP = 0.6
TOP_K = 60

class OnionHuggingFaceLM():
    def __init__(self, layers, is_last=False):
        self.layers = layers
        self.is_last = is_last

    def forward(
            self,
            model,
            input_ids: torch.tensor=None,
            hidden_states: torch.tensor=None,
            attention_mask: torch.tensor=None,
            **kwargs
        ) -> Tuple[Optional[torch.tensor], Optional[torch.tensor]]:

        # set base model
        base_model = model.model

        if input_ids is not None and hidden_states is not None:
            print("You must either pass a hidden_state or input_ids but not both")
            assert ValueError

        if input_ids is not None:
            # embed
            hidden_states = base_model.embed_tokens(input_ids)
            position_ids = torch.arange(
                0,
                input_ids.size(1),
                device=input_ids.device
            ).unsqueeze(0)

        if hidden_states is not None:
            hidden_states = hidden_states
            position_ids = torch.arange(
                0,
                hidden_states.size(1),
                device=hidden_states.device
            ).unsqueeze(0)

        for layer in self.layers:
            print(f"Processing hidden state from layer\n{layer}\n")
            hidden_states = layer(
                hidden_states,
                position_ids=position_ids
            )[0]

        if self.is_last:
            norm_states = base_model.norm(hidden_states).to("cuda")
            logits = model.lm_head(norm_states).to("cuda")

            return (None, logits)
        
        return (hidden_states, None)

async def model_half_split_test(prompt: str, model_id: str, layers: int):
    """
    Test for splitting in half
    """

    half_layers = int(layers / 2)

    # inference
    tokenizer = await resolve_tokenizer(model_id)
    max_length = 512 #tokenizer.model_max_length

    # get full model
    if re.match(r"^Qwen|qwen", model_id):
        model = Qwen2ForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            # attn_implementation="eager"
            # low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            # low_cpu_mem_usage=True
        )

    print(model.hf_device_map)

    # add pad token if none, depending on model
    #if tokenizer.pad_token == None:
    #    if re.match(r"Llama|llama", model_id):
    #        tokenizer.add_special_tokens({"pad_token":"<pad>"})
    #        model.resize_token_embeddings(len(tokenizer))

    shard_layers = nn.ModuleList(model.model.layers[:half_layers])#.to("cuda")
    sharded_model = OnionHuggingFaceLM(layers=shard_layers)

    print(model)

    # generate first half
    messages = [{"role": "user", "content": prompt}]
    txt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"Generating from chat template\n{txt}")

    inputs = tokenizer([txt], return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    input_attention_mask = inputs.attention_mask.to("cuda")

    # add if first layer of model check
    shard_hidden_states, shard_logits = sharded_model.forward(
        model=model,
        input_ids=input_ids
    )

    print(f"shard_hidden_states\n{shard_hidden_states}")
    print(f"shard_logits\n{shard_logits}")


    # second half
    print("Using first half hidden state for last half of model")
    shard_layers = nn.ModuleList(model.model.layers[half_layers:]).to("cuda")
    sharded_model.layers = shard_layers
    sharded_model.is_last = True 

    if shard_hidden_states is not None:
        # add if last layer of model or in the middle check
        shard_hidden_states, shard_logits = sharded_model.forward(
            model=model,
            hidden_states=shard_hidden_states
        )

        print(f"shard_hidden_states\n{shard_hidden_states}")
        print(f"shard_logits\n{shard_logits}")
    else:
        print("Sharded hidden states not found, error")
        raise ValueError
    

    print("generate from logits")
    if shard_logits is not None:
        print(shard_logits.dim())
        #print(shard_logits[0])

        generated_ids = sample_logits(shard_logits, 0.1, 0.95, 30)
        #generated_ids = torch.argmax(shard_logits/0.7, dim=-1)
        #generated_ids = model.generate(logits)
        
        print("generated_ids")
        print(generated_ids)

        generated_text = tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

        print("Generated text:")
        print(generated_text)
    else:
        print("Sharded logits missing from last layer run, error")
        raise ValueError

    # free model from memory
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
   prompt = "In a single word only, what is the last name of the current president of the USA?"

   print("\n-------- Test TinyLlama/TinyLlama_v1.1 ----------\n")
   model_id = "TinyLlama/TinyLlama_v1.1"
   model_layers = 22

   asyncio.run(
       model_half_split_test(
           prompt=prompt,
           model_id=model_id,
           layers=model_layers
       )
   )

    #print("\n-------- Test meta-llama/Meta-Llama-3.1-8B ----------\n")
    #model_id = "meta-llama/Meta-Llama-3.1-8B"
    #model_layers = 32

    #asyncio.run(
    #    model_half_split_test(
    #        prompt=prompt,
    #        model_id=model_id,
    #        layers=model_layers
    #    )
    #)

   #print("\n-------- Test Qwen/Qwen2-57B-A14B-Instruct ----------\n")
   #model_id = "Qwen/Qwen2-57B-A14B-Instruct"
   #model_layers = 28

   #asyncio.run(
   #    model_half_split_test(
   #        prompt=prompt,
   #        model_id=model_id,
   #        layers=model_layers
   #    )
   #)

