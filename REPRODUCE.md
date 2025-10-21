- install zlib-dev as dependencies for triton
- install uv
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
- install pytorch==2.1.0 torchvision==0.16.0
- create venv with python = 3.10
```
uv pip install pytorch==2.1.0 torchvision==0.16.0
uv sync
```


config.yml
```yml
# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================
exp_name: "yochameleon_baseline"
project_name: "yochameleon"
entity: null  # Your WandB entity/username
savedir: "./checkpoints"
no_wandb: false

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================
model_id: "leloy/Anole-7b-v0.1-hf"  # or "Emu3-community/Emu3-Gen-hf"
whole_model: false  # If true, train entire model; if false, train only embeddings and lm_head

# ==============================================================================
# SUBJECT/PERSONALIZATION CONFIGURATION
# ==============================================================================
sks_name: "SKS_NAME"  # Will be replaced if --sks_name is provided via CLI
json_file:
  - "data/SKS_NAME/metadata_train.json"
  - "data/SKS_NAME/metadata_val.json"

### Download data

```
# Mini ChameLeon
git lfs install
git clone git@huggingface.co:datasets/thaoshibe/Mini-YoChameleon-Data

# for YoLLaVA data
git lfs install
git clone https://huggingface.co/datasets/thaoshibe/YoLLaVA
```

### Missing code

- Self prompting
commit: 6b08ce4f67fef36b2afb3bdfd571c6dabfdd3d6b

- yochameleon.py Yochameleon Trainer: train_epoch_disjoin, train_api, finetune_epoch, train, finetune
  - current: only train_epoch remains 
last commit: 6aef5d1