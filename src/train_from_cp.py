# Huggingface is annoying
from src.init_hf_path import init_hf_path
init_hf_path()

import torch
from src.Trainer import Trainer
import click
from typing import Optional



@click.command()
@click.option("--model_save_path", required=True, type=str)
@click.option("--step", required=False, type=int, default=None, help="Step to restart from. Defaults to the highest checkpoint step in the given `model_save_path`.")
@click.option("--reset_wandb", required=False, is_flag=True, help="Add flag to start a new wandb run under the `new_wandb_name` name. No flag to continue from the wandb name the checkpoint is tied to.")
@click.option("--new_wandb_name", required=False, type=str, default=None, help="If reset_wandb is True, this name is used for the new wandb run. Defaults to the name of the saved run.")
@click.option("--num_devices", required=False, type=int, default=None, help="Number of devices to run on. Defaults to whatever was used for training.")
def main(
        model_save_path: str,
        step: Optional[int],
        reset_wandb: bool,
        new_wandb_name: Optional[str],
        num_devices: Optional[int],
    ):
    
    # Loading in a model and training from that cp
    Trainer.load_checkpoint(
        model_save_path=model_save_path,
        step=step,
        reset_wandb=reset_wandb,
        new_wandb_name=new_wandb_name,
        num_devices=num_devices,
    )()




if __name__ == "__main__":
    main()
