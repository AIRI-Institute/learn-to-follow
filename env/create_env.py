import time

import numpy as np
from pogema import AnimationConfig, AnimationMonitor

from pogema import pogema_v0

from follower.training_config import Environment

import gymnasium
import re
from copy import deepcopy
from pogema import GridConfig

from env.custom_maps import MAPS_REGISTRY
from follower.preprocessing import wrap_preprocessors, PreprocessorConfig


class ProvideGlobalObstacles(gymnasium.Wrapper):
    def get_global_obstacles(self):
        return self.grid.get_obstacles().astype(int).tolist()

    def get_global_agents_xy(self):
        return self.grid.get_agents_xy()


def create_env_base(config: Environment):
    env = pogema_v0(grid_config=config.grid_config)
    env = ProvideGlobalObstacles(env)
    if config.use_maps:
        env = MultiMapWrapper(env)
    if config.with_animation:
        env = AnimationMonitor(env, AnimationConfig(directory='renders',
                                                    egocentric_idx=None,
                                                    show_lines=True))

    # adding runtime metrics
    env = RuntimeMetricWrapper(env)

    return env


class RuntimeMetricWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._start_time = None
        self._env_step_time = None

    def step(self, actions):
        env_step_start = time.monotonic()
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        env_step_end = time.monotonic()
        self._env_step_time += env_step_end - env_step_start
        if all(terminated) or all(truncated):
            final_time = time.monotonic() - self._start_time - self._env_step_time
            if 'metrics' not in infos[0]:
                infos[0]['metrics'] = {}
            infos[0]['metrics'].update(runtime=final_time)
        return observations, rewards, terminated, truncated, infos

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._start_time = time.monotonic()
        self._env_step_time = 0.0
        return obs


class MultiMapWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._configs = []
        self._rnd = np.random.default_rng(self.grid_config.seed)
        pattern = self.grid_config.map_name

        if pattern:
            for map_name in sorted(MAPS_REGISTRY):
                if re.match(pattern, map_name):
                    cfg = deepcopy(self.grid_config)
                    cfg.map = MAPS_REGISTRY[map_name]
                    cfg.map_name = map_name
                    cfg = GridConfig(**cfg.dict())
                    self._configs.append(cfg)
            if not self._configs:
                raise KeyError(f"No map matching: {pattern}")

    def reset(self, seed=None, **kwargs):
        self._rnd = np.random.default_rng(seed)
        if self._configs is not None and len(self._configs) >= 1:
            map_idx = self._rnd.integers(0, len(self._configs))
            cfg = deepcopy(self._configs[map_idx])
            self.env.unwrapped.grid_config = cfg
            self.env.unwrapped.grid_config.seed = seed
        return self.env.reset(seed=seed, **kwargs)


def main():
    env = create_env_base(config=Environment())
    env = wrap_preprocessors(env, config=PreprocessorConfig())
    env.reset()
    env.render()


if __name__ == '__main__':
    main()
