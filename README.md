<div align="center">

[![Example](https://raw.githubusercontent.com/Tviskaron/pogema-svg/main/learn-to-follow-ep00001-lab-maze_010-seed0.svg)](https://github.com/AIRI-Institute/learn-to-follow) 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CnC47qbc4Z3sHfiR6sIX0ngXi6UfTx8o?usp=sharing)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/AIRI-Institute/learn-to-follow/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2310.01207-b31b1b.svg)](https://arxiv.org/abs/2310.01207)
[![Paper](https://img.shields.io/badge/AAAI-2024-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/29704)

**Learn to Follow: Lifelong Multi-agent Pathfinding with Decentralized Replanning**

</div> 

This study addresses the challenging problem of decentralized lifelong multi-agent pathfinding. The proposed **Follower** 
approach utilizes a combination of a planning algorithm for constructing a long-term plan and reinforcement learning
for resolving local conflicts.

**Paper:** [Learn to Follow: Decentralized Lifelong Multi-agent Pathfinding via Planning and Learning
](https://arxiv.org/abs/2310.01207)



## Installation:

```bash
pip3 install -r docker/requirements.txt
```


Installation of ONNX runtime:
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.14.1/onnxruntime-linux-x64-1.14.1.tgz \
    && tar -xf onnxruntime-linux-x64-1.14.1.tgz \
    && cp onnxruntime-linux-x64-1.14.1/lib/* /usr/lib/ && cp onnxruntime-linux-x64-1.14.1/include/* /usr/include/
```

Optionally, you could use the Dockerfile to build the image:
```bash
cd docker && sh build.sh
```

## Inference Example:

To execute the **Follower** algorithm and produce an animation using pre-trained weights, use the following command:

```bash
python3 example.py
```

The animation will be stored in the `renders` folder.

It's recommended to set environment variable to restrict Numpy CPU threads to 1,  avoiding performance issues:

```bash
export OMP_NUM_THREADS="1" 
export MKL_NUM_THREADS="1" 
export OPENBLAS_NUM_THREADS="1"
```

You can adjust the environment and algorithm parameter using arguments. For example:
```
python3 example.py --map_name wfi_warehouse --num_agents 128
python3 example.py --map_name pico_s00_od20_na32 --num_agents 32 --algorithm FollowerLite
```


We offer a Google Colab example that simplifies the process:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CnC47qbc4Z3sHfiR6sIX0ngXi6UfTx8o?usp=sharing)


## Training:

To train **Follower** from scratch, use the following command:

```bash
python3 main.py  --actor_critic_share_weights=True --batch_size=16384 --env=PogemaMazes-v0 --exploration_loss_coeff=0.023 --extra_fc_layers=1 --gamma=0.9756 --hidden_size=512 --intrinsic_target_reward=0.01 --learning_rate=0.00022 --lr_schedule=constant --network_input_radius=5 --num_filters=64 --num_res_blocks=8 --num_workers=8 --optimizer=adam --ppo_clip_ratio=0.2   --train_for_env_steps=1000000000 --use_rnn=True
```

To train **FollowerLite** from scratch, use the following command:
```bash
python3 main.py  --actor_critic_share_weights=True --batch_size=16384 --env=PogemaMazes-v0 --exploration_loss_coeff=0.0156 --extra_fc_layers=0 --gamma=0.9716 --hidden_size=16 --intrinsic_target_reward=0.01 --learning_rate=0.00013 --lr_schedule=kl_adaptive_minibatch --network_input_radius=3 --num_filters=8 --num_res_blocks=1 --num_workers=4 --optimizer=adam --ppo_clip_ratio=0.2     --train_for_env_steps=20000000 --use_rnn=False
```
The parameters are set to the values used in the paper.

### Testing and Results Visualization 
To reproduce the main results of **Follower** and **FollowerLite** using [pogema-toolbox](https://github.com/AIRI-Institute/pogema-toolbox), use the following command:
```bash
python3 eval.py
```
This script will run all the experiments, the configurations for which are placed in the experiments folder. The raw data will be saved in the corresponding folders (including plots) and optionally saved to wandb.

#### Example Configuration:

```yaml
environment:
  name: Pogema-v0
  on_target: restart
  max_episode_steps: 512
  observation_type: POMAPF
  collision_system: soft  
  map_name: wfi_warehouse
  num_agents:
    grid_search: [ 32, 64, 96, 128, 160, 192 ]
  seed:
    grid_search: [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]

algorithms:
  Follower:
    name: Follower

  No dynamic cost:
    name: Follower
    override_config:
      preprocessing:
        use_dynamic_cost: False

  No static cost:
    name: Follower
    override_config:
      preprocessing:
        use_static_cost: False

results_views:
  TabularResults:
    type: tabular
    drop_keys: [ seed ]
    print_results: True

  05-warehouse:
    type: plot
    x: num_agents
    y: avg_throughput
    name: Warehouse $46 \times 33$
```

#### Description of Configuration:

The configuration defines the environment settings and the algorithms used for the experiments. It specifies the following:
- **Environment**: Includes parameters of the POGEMA environment, behavior on target (restart, corresponding to LifeLong), maximum episode steps (512), observation type, collision system, etc. It also sets up grid searches for the number of agents and seed values. The `grid_search` can be used for any environment parameter.
- **Algorithms**: Details the algorithms to be tested. The primary algorithm is **Follower**. Variants include "No dynamic cost" and "No static cost," which override specific preprocessing configurations.
- **Results Views**: Defines how the results will be presented, including tabular and plot views.

This example configuration demonstrates how to set up experiments for the Pogema-v0 environment, varying the number of agents and seeds, and comparing different versions of the Follower algorithm.
#### Raw Data

The raw data, comprising the results of our experiments for Follower and FollowerLite, can be downloaded from the following link:
[Download Raw Data](https://github.com/AIRI-Institute/learn-to-follow/releases/download/v0/learn-to-follow-raw-data.zip)


## Citation:

```bibtex
@inproceedings{skrynnik2024learn,
  title={Learn to Follow: Decentralized Lifelong Multi-Agent Pathfinding via Planning and Learning},
  author={Skrynnik, Alexey and Andreychuk, Anton and Nesterova, Maria and Yakovlev, Konstantin and Panov, Aleksandr},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={16},
  pages={17541--17549},
  year={2024}
}
```

