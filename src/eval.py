# Huggingface is annoying
from src.init_hf_path import init_hf_path
init_hf_path()
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

import torch
from src.model.Model import load_model_for_inf
from typing import Optional
import lm_eval
from lm_eval.models.huggingface import HFLM
import click
from dataclasses import dataclass
from src.types import InfMode
from src.configs import InfConfig

# Hold all info on an eval
@dataclass
class EvalInfo():
    # Name of the eval
    name: str
    # Which scores to read for the eval
    score_names: list[str]
    # Number of shots for the eval
    num_shots: int
    # Which type of task is this?
    task_type: InfMode

# Registered evals
registered_evals = {
    "wikitext": EvalInfo(
        name="WikiText",
        score_names=["word_perplexity,none"],
        num_shots=0,
        task_type=InfMode.PREFILL,
    ),
    "lambada_openai": EvalInfo(
        name="lmb",
        score_names=["acc,none"],
        num_shots=0,
        task_type=InfMode.PREFILL,
    ),
    "swde": EvalInfo(
        name="SWDE",
        score_names=["contains,none"],
        num_shots=0,
        task_type=InfMode.DECODE,
    ),
    "squad_completion": EvalInfo(
        name="SQD",
        score_names=["contains,none"],
        num_shots=0,
        task_type=InfMode.DECODE,
    ),
    "fda": EvalInfo(
        name="FDA",
        score_names=["contains,none"],
        num_shots=0,
        task_type=InfMode.DECODE,
    ),
    "triviaqa": EvalInfo(
        name="TQA",
        score_names=["exact_match,none"],
        num_shots=5,
        task_type=InfMode.DECODE,
    ),
    "nq_open": EvalInfo(
        name="NQ",
        score_names=["exact_match,none"],
        num_shots=5,
        task_type=InfMode.DECODE,
    ),
    "drop": EvalInfo(
        name="DROP",
        score_names=["f1,none"],
        num_shots=3,
        task_type=InfMode.DECODE,
    ),
    "hellaswag": EvalInfo(
        name="HellaSwag",
        score_names=["acc_norm,none"],
        num_shots=0,
        task_type=InfMode.PREFILL,
    ),
    "piqa": EvalInfo(
        name="PIQA",
        score_names=["acc_norm,none"],
        num_shots=0,
        task_type=InfMode.PREFILL,
    ),
    "arc_easy": EvalInfo(
        name="ARC-easy",
        score_names=["acc_norm,none"],
        num_shots=0,
        task_type=InfMode.PREFILL,
    ),
    "arc_challenge": EvalInfo(
        name="ARC-challenge",
        score_names=["acc_norm,none"],
        num_shots=0,
        task_type=InfMode.PREFILL,
    ),
    "race": EvalInfo(
        name="RACE-high",
        score_names=["acc,none"],
        num_shots=5,
        task_type=InfMode.PREFILL,
    ),
    "humaneval": EvalInfo(
        name="HumanEval",
        score_names=["pass@1,create_test"],
        num_shots=0,
        task_type=InfMode.DECODE,
    ),
    "mbpp": EvalInfo(
        name="MBPP",
        score_names=["pass_at_1,none"],
        num_shots=3,
        task_type=InfMode.DECODE,
    ),
    "triviaqa": EvalInfo(
        name="TriviaQA",
        score_names=["exact_match,remove_whitespace"],
        num_shots=5,
        task_type=InfMode.DECODE,
    ),
    "nq_open": EvalInfo(
        name="NaturalQuestions",
        score_names=["exact_match,remove_whitespace"],
        num_shots=5,
        task_type=InfMode.DECODE,
    ),
}




@torch.no_grad()
@click.command()
@click.option("--model_save_path", required=True, type=str)
@click.option("--eval_names", required=True, type=str, help="Comma seperated list of eval names to run")
@click.option("--step", required=False, type=int, default=None, help="Step to load in. Defaults to the highest checkpoint step in the given `model_save_path`.")
@click.option("--device", required=False, type=str, default="cuda:0", help="device to load model. Defaults to cuda:0")
def main(
        model_save_path: str,
        eval_names: str,
        step: Optional[int],
        device: Optional[str],
    ):
    # Get eval names. Ensure they all exist
    eval_names = eval_names.strip()
    if eval_names[-1] == ",":
        eval_names = eval_names[:-1]
    eval_names = eval_names.replace(", ", ",")
    eval_names = eval_names.split(",")
    for eval_name in eval_names:
        assert eval_name in registered_evals
    
    # Load in the model
    model = load_model_for_inf(
        model_save_path=model_save_path,
        step=step,
        device=device
    )
    
    # Stuff for HFLM
    @dataclass
    class Output():
        logits: torch.Tensor
    class ModelWrapper():
        def __init__(self, model):
            self.model = model
            self.device = torch.device(device)
            self.config = {}
            def fn():
                return
            self.tie_weights = fn

        def __call__(self, input_ids):
            # Not decoding, just a fwd pass
            self.model.decoding = False
            
            # Need to turn input_ids into a flattened
            # sequence with cu_seqlens
            # (B, N) -> (B*N)
            B, N = input_ids.shape
            input_ids_ = input_ids.flatten(0, 1)
            cu_seqlens = torch.tensor(
                [i*N for i in range(0, B+1)],
                device=input_ids.device,
                dtype=torch.long
            )
            out = self.model(
                input_ids_,
                cu_seqlens,
                return_loss=False
            )
            return Output(
                logits=out.reshape(B, N, -1)
            )
            
        def generate(
                self,
                input_ids,
                max_length,
                stopping_criteria,
                pad_token_id,
                use_cache,
                **generation_kwargs,
            ):
            sampling_strategy = "greedy"
            
            # Seqlens of each input
            B, N = input_ids.shape
            cu_seqlens = torch.cat((
                torch.tensor(
                    [0],
                    device=input_ids.device,
                    dtype=torch.long
                ),
                (input_ids != pad_token_id).sum(-1).long()
            ), dim=0).cumsum(0)
            
            # Starting and ending positions for
            # each input and if they are done decoding
            starts = [(row == pad_token_id).cumprod(0).sum().item() for row in input_ids]
            ends = [N]*B
            done = [False]*B
            
            # Allocate tensor for output sequences
            outputs = torch.empty(
                B, max_length,
                device=input_ids.device,
                dtype=input_ids.dtype,
            ).fill_(self.model.tokenizer.eos_token_id)
            outputs[:, :N] = input_ids

            # First iter config
            inf_cfg = InfConfig(
                # Doing prefill
                inf_mode=InfMode.PREFILL,
                # We want to cache KVs
                update_cache=[True]*B,
                # Reset cache from previous fwd pass
                reset_cache=True
            )
            
            # Decode loop
            while False in done:
                # Flatten inputs
                input_ids_ = (
                    torch.stack([
                        outputs[b, ends[b] - 1]
                        for b in range(0, B)
                    ], dim=0)
                    if inf_cfg.inf_mode == InfMode.DECODE
                    else torch.cat([
                        outputs[b, starts[b]:ends[b]]
                        for b in range(0, B)
                    ], dim=0)
                )
                
                # Forward pass
                out = self.model(
                    input_ids_,
                    cu_seqlens,
                    return_loss=False,
                    inf_cfg=inf_cfg,
                )
                
                # Get next token probs for each seq
                # (B, V)
                if inf_cfg.inf_mode == InfMode.PREFILL:
                    out = torch.stack([
                        out[cu_seqlens[b + 1] - 1]
                        for b in range(0, B)
                    ], dim=0)

                # Evaluate the stopping criteria
                for sc in stopping_criteria:
                    # Evaluate stopping criteria
                    sc(outputs[:, :max(ends)], out)
                    # Update which sequences are done
                    for b, flag in enumerate(getattr(sc, "done_tracker", [])):
                        done[b] = done[b] or flag
                
                # Sample next token
                if sampling_strategy == "sample":
                    next_toks = torch.distributions.Categorical(
                        probs = out.softmax(-1)
                    ).sample()
                elif sampling_strategy == "greedy":
                    next_toks = out.argmax(-1)
                else:
                    raise NotImplementedError
                
                # Add tokens, increase seqlens
                # and increase ends
                for b in range(0, B):
                    # Skip already ended sequences
                    if done[b]:
                        continue
                    
                    outputs[b, ends[b]] = next_toks[b]
                    ends[b] += 1

                    # First sequence increases by 1, second
                    # increases by 2 (needs to include the
                    # increase to the first seq) and so on
                    cu_seqlens[b+1:] += 1
                    
                    # End on an eos token or if the seq
                    # reached the seq limit
                    done[b] = (
                        next_toks[b] == self.model.tokenizer.eos_token_id
                        or ends[b] == max_length
                    )

                # Now doing decode and want to keep
                # the cache
                inf_cfg = InfConfig(
                    inf_mode=InfMode.DECODE,
                    # Don't update the KV cache for done sequences
                    update_cache=[not d for d in done],
                    reset_cache=False
                )
                
            
            return outputs
    
    def run_eval(eval_name):
        eval_info = registered_evals[eval_name]
        name = eval_info.name
        score_names = eval_info.score_names
        num_shots = eval_info.num_shots
        task_type = eval_info.task_type
        
        seqlen = model.max_seqlen
        bs = 4 if task_type == InfMode.PREFILL else 16
        print(f"eval as seqlen {seqlen} and bs {bs}")
        
        results = lm_eval.simple_evaluate(
            model=HFLM(
                pretrained=ModelWrapper(model),
                tokenizer=model.tokenizer,
                batch_size=bs,
                max_length=seqlen,
            ),
            tasks=[eval_name],
            bootstrap_iters=0,
            num_fewshot=num_shots,
            confirm_run_unsafe_code=True,
            # samples={eval_name: [0, 1, 2, 3, 4]}
        )
        
        result = results["results"][eval_name]
        res = {
            score_name: result[score_name]
            for score_name in score_names
        }
        print(eval_name, res)
        return res
        
    results = []
    for eval_name in eval_names:
        results.append(run_eval(eval_name))
        
    for result, eval_name in zip(results, eval_names):
        print(f"eval name: {eval_name}:")
        print(result)




if __name__ == "__main__":
    main()
