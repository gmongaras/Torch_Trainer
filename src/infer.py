# Huggingface is annoying
from src.init_hf_path import init_hf_path
init_hf_path()

import torch
from typing import Optional
import click
from src.model.Model import load_model_for_inf
from tqdm import tqdm





@torch.no_grad()
@click.command()
@click.option("--prompt", required=True, type=str)
@click.option("--model_save_path", required=True, type=str)
@click.option("--step", required=False, type=int, default=None, help="Step to load in. Defaults to the highest checkpoint step in the given `model_save_path`.")
@click.option("--device", required=False, type=str, default="cuda:0", help="device to load model. Defaults to cuda:0")
@click.option("--num_decode_steps", required=False, type=int, default=None, help="Max of steps to decode for.")
@click.option("--early_stop", required=False, is_flag=True, default=None, help="Add flag to stop at the first decoded stop token.")
@click.option("--sampling_strategy", required=False, type=str, default="sample", help="How to sample the next token. Can be one of `sample` or `greedy`")
def main(
        prompt: str,
        model_save_path: str,
        step: Optional[int],
        device: Optional[str],
        num_decode_steps: Optional[int],
        early_stop: bool,
        sampling_strategy: str,
    ):
    assert sampling_strategy in ["sample", "greedy"]
    
    # Load in the model
    print("Loading model...")
    model = load_model_for_inf(
        model_save_path=model_save_path,
        step=step,
        device=device
    )
    print("Model loaded")
    
    # Tokenize prompt
    input_ids = model.tokenizer.encode(
        prompt,
        add_special_tokens=True
    )
    cu_seqlens = torch.tensor(
        [0, len(input_ids)],
        dtype=torch.long,
        device=device,
    )
    
    
    # Decode loop
    print("Generating...")
    num_dec = 0
    with tqdm(total=num_decode_steps, desc="Processing") as pbar:
        while True:
            # Model fwd pass
            logits = model.forward(
                inputs=torch.tensor(
                    input_ids,
                    dtype=torch.long,
                    device=device,
                ), 
                cu_seqlens=cu_seqlens,
                return_loss=False,
            )
            
            # Sample next token
            if sampling_strategy == "sample":
                next_tok = torch.distributions.Categorical(
                    probs = logits.softmax(-1)
                ).sample()
            elif sampling_strategy == "greedy":
                next_tok = logits.argmax(-1)
            else:
                raise NotImplementedError
            
            # Add to cur seq
            input_ids.append(next_tok)
            cu_seqlens[-1] += 1
            
            
            num_dec += 1
            
            # Stop decoding if we hit a stop token
            # or have decoded to the decode limit
            if num_decode_steps is not None and num_dec > num_decode_steps:
                break
            if early_stop and next_tok == model.tokenizer.eos_token_id:
                break

            pbar.update(1)
    
    print(model.tokenizer.decode(input_ids))
    




if __name__ == "__main__":
    main()
