from pogema_toolbox.create_env import create_env_base, Environment
from pogema_toolbox.evaluator import evaluation
from pogema import BatchAStarAgent

from pogema_toolbox.eval_utils import initialize_wandb, save_evaluation_results, create_and_push_summary_archive

from pathlib import Path
import wandb

import yaml

from pogema_toolbox.registry import ToolboxRegistry

from follower.inference import FollowerInference, FollowerInferenceConfig
from follower.preprocessing import follower_preprocessor
from follower_cpp.inference import FollowerConfigCPP, FollowerInferenceCPP
from follower_cpp.preprocessing import follower_cpp_preprocessor

PROJECT_NAME = 'pogema-toolbox'
BASE_PATH = Path('experiments')


def main(disable_wandb=True):
    ToolboxRegistry.register_env('Pogema-v0', create_env_base, Environment)
    ToolboxRegistry.register_algorithm('A*', BatchAStarAgent)
    ToolboxRegistry.register_algorithm('Follower', FollowerInference, FollowerInferenceConfig, follower_preprocessor)

    ToolboxRegistry.register_algorithm('FollowerLite', FollowerInferenceCPP, FollowerConfigCPP,
                                       follower_cpp_preprocessor)

    with open("env/test-maps.yaml", 'r') as f:
        maps_to_register = yaml.safe_load(f)
    ToolboxRegistry.register_maps(maps_to_register)

    folder_names = [
        '01-random-20x20',
    ]

    for folder in folder_names:
        config_path = BASE_PATH / folder / f"{Path(folder).name}.yaml"
        eval_dir = BASE_PATH / folder

        with open(config_path) as f:
            evaluation_config = yaml.safe_load(f)
        if folder == 'eval-fast':
            disable_wandb = True

        initialize_wandb(evaluation_config, eval_dir, disable_wandb, PROJECT_NAME)
        evaluation(evaluation_config, eval_dir=eval_dir)
        save_evaluation_results(eval_dir)
        wandb.finish()


if __name__ == '__main__':
    main()
