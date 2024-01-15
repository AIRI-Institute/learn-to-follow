from env.create_env import create_env_base
from env.warehouse_wfi import create_warehouse_wfi_env
from utils.eval_utils import run_episode
from follower.training_config import EnvironmentMazes
from follower.inference import FollowerInferenceConfig, FollowerInference
from follower.preprocessing import follower_preprocessor
from follower_cpp.inference import FollowerConfigCPP, FollowerInferenceCPP
from follower_cpp.preprocessing import follower_cpp_preprocessor


def create_warehouse_env(num_agents=128, seed=0):
    env_cfg = EnvironmentMazes(with_animation=True, )
    env_cfg.grid_config.num_agents = num_agents
    env_cfg.grid_config.seed = seed
    return create_warehouse_wfi_env(env_cfg)


def create_custom_env(map_name='test-mazes-s41_wc5_od50', num_agents=256, seed=0):
    env_cfg = EnvironmentMazes(with_animation=True)
    env_cfg.grid_config.num_agents = num_agents
    env_cfg.grid_config.map_name = map_name
    env_cfg.grid_config.seed = seed
    return create_env_base(env_cfg)


def run_follower(env):
    follower_cfg = FollowerInferenceConfig()
    algo = FollowerInference(follower_cfg)

    env = follower_preprocessor(env, follower_cfg)

    print(run_episode(env, algo))


def run_follower_cpp(env):
    follower_cfg = FollowerConfigCPP(path_to_weights='model/follower-lite', num_threads=6)
    algo = FollowerInferenceCPP(follower_cfg)

    env = follower_cpp_preprocessor(env, follower_cfg)

    print(run_episode(env, algo))


def main():
    run_follower(create_warehouse_env())
    run_follower(create_custom_env(map_name="test-mazes-s41_wc5_od50"))


if __name__ == '__main__':
    main()
