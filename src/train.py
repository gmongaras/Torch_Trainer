# Huggingface is annoying
from src.init_hf_path import init_hf_path
init_hf_path()

from distributed_helpers import is_main_process
from src.Trainer import Trainer
from src.exp_configs import (
    TrainerConfigs,
    ModelConfigs,
    OptimConfigs,
    get_config
)
import click



@click.command()
@click.option("--model_config_name", required=True, type=str)
@click.option("--optim_config_name", required=True, type=str)
@click.option("--trainer_config_name", required=True, type=str)
def main(
        model_config_name: str,
        optim_config_name: str,
        trainer_config_name: str,
    ):
    # Load in configs
    model_config = get_config(ModelConfigs, model_config_name)
    optim_config = get_config(OptimConfigs, optim_config_name)
    trainer_config = get_config(TrainerConfigs, trainer_config_name)
    
    if is_main_process():
        print(
f"""
-------------------------------------
Running with the following setup:
Model config: {model_config_name}:
{model_config}

Optim config: {optim_config_name}
{optim_config}

Trainer config: {trainer_config_name}
{trainer_config}
-------------------------------------
""".strip()
)
    
    # Run the trainer
    Trainer(
        model_config=model_config,
        optim_config=optim_config,
        trainer_config=trainer_config,
    )()




if __name__ == "__main__":
    main()
