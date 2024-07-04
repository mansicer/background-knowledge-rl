# Sample Efficient RL with Background Knowledge

Here is the code repo of the paper "Improving Sample Efficiency of Reinforcement Learning with Background Knowledge from Large Language Models".

## Installation
You can create a Python virtual environment through `conda`. Then install the necessary packages from the following script.
```
bash install.sh
```
Afterward, install the two environments with text support. We adopt the captioners directly from [GLAM](https://github.com/flowersteam/Grounding_LLMs_with_online_RL) in Minigrid and [SmartPlay](https://github.com/microsoft/SmartPlay) in Crafter.
```
pip install -e minigrid_text
pip install -e crafter_text
```

## Run Experiments

### Data Collection
We provide our pre-collected data in the Minigrid environment in the `data/Minigrid-dataset` folder, which can be directly used for below precedures. We do not supplement data in the Crafter due to the storage issue. Therefore, we also provide the command to collect the dataset manually: 
```bash
python train_rnd.py --env <env> --precollect --num_eval_episodes 50 --eval_freq 500000 --total_timesteps 5000000
```
Here `<env>` can be `BabyAI-Text-GoToLocal-Pretraining-S20` for Minigrid and `Crafter-Text-Reward` for Crafter. The collected data will be stoned in the `data` folder by default.

### Background Knowledge Representation
We propose three variants, BK-Code, BK-Pref, and BK-Goal, which have different prompting mechanisms. We implement the prompting processes through three separate files. Please set up the LLM API configs before running the code.

#### LLM API setup
For our experiments, we use the OpenAI `gpt-3.5-turbo-1106` and `gpt-4-0613` models through the Azure OpenAI API. You should prepare the config file in JSON format placed at the `cache` folder based on your account. Our programs will read the config in this folder by matching the `model_name` argument. The LLM call logic is implemented in `utils/azure_llm.py`. You can adapt the implementation to OpenAI API or a customized language server by modifying functions inside.

#### BK-Code
```bash
python label_code.py --dataset_path data/Minigrid-dataset --env minigrid --model_name gpt-4
```
The command above will prompt LLMs to write code and save the results in `logs/bk-code`.

#### BK-Pref
```bash
python label_preference.py --dataset_path data/Minigrid-dataset --env minigrid --model_name gpt-4 --dataset_size 5000
```
The command above will prompt LLMs to annotate preference from sampled data. The `env` can be changed to `crafter` with a Crafter dataset. The results will be saved to the `data` path starting with the LLM model name. After annotating the preference, we need to train a parameterized potential function model. 
```bash
python train_reward_model.py --dataset_path <annotation-data-path> --env minigrid
```
The command will save model checkpoints in `logs/bk-pref`.

#### BK-Goal
```bash
python label_goals.py --dataset_path=data/Minigrid-dataset --env minigrid --model_name gpt-4
```
The results will be saved to `logs/bk-goal`.

### Run Downstream RL tasks
Using results from background knowledge representation, we can run RL tasks with reward shaping using LLM knowledge.
```bash
python train_rs.py --alg <bk-code|bk-pref|bk-goal> --pretrain_path <logs/xxx/run_name> --env BabyAI-Text-GoToLocal-RedBall-S20
```
The `alg_note` can be `bk-code`, `bk-pref`, and `bk-goal` according to the algorithm. You should also replace the `pretrain_path` with correct folder. The environment can be replaced with any registration from `babyai_envs/__init__.py` and `crafter_text/__init__.py`. The results will be logged in `logs/rl/<alg_name>` with tensorboard and configurable wandb logs (please refer to `utils/config.py`). 

We also provide some example runs of background knowledge representation, with which you can directly run an RL algorithm to omit the pervious stages:
```bash
# BK-Code
python train_rs.py --alg bk-code --pretrain_path logs/bk-code/gpt-4-sample --env BabyAI-Text-GoToLocal-RedBall-S20

# BK-Pref
python train_rs.py --alg bk-pref --pretrain_path logs/bk-pref/gpt-4-sample --env BabyAI-Text-GoToLocal-RedBall-S20

# BK-Goal
python train_rs.py --alg bk-goal --pretrain_path logs/bk-goal/gpt-4-sample --env BabyAI-Text-GoToLocal-RedBall-S20
```
