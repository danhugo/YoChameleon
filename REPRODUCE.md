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

# ==============================================================================
# TOKENIZER CONFIGURATION
# ==============================================================================
tokenizer_max_length: 4096  # Maximum sequence length for tokenizer

# ==============================================================================
# SPECIAL TOKENS CONFIGURATION
# ==============================================================================
special_tokens:
  SKS_TOKEN: "<sks>"          # Identifier token
  LATENT_TOKEN_START: 16200   # Starting index for latent tokens (e.g., <reserved16200>)
  START_OF_IMAGE_INDEX: 8710  # Token ID for start of image
  END_OF_IMAGE_INDEX: 8711    # Token ID for end of image
  END_OF_TURN: 8706           # Token ID for end of turn (<reserved08706>)

# ==============================================================================
# PERSONALIZED TOKEN CONFIGURATION
# ==============================================================================
prefix_token: 3  # Number of prefix tokens to use (e.g., 3 tokens per task)
different_identifier: false  # Use different identifiers for negative samples (deprecated)

# ==============================================================================
# SELF-PROMPTING CONFIGURATION
# ==============================================================================
self_prompting: false  # Enable self-prompting with separate generation/understanding tokens
# If self_prompting is true:
#   - Personalized tokens: <sks> + generation_tokens + understanding_tokens
#   - Prompt format: "<sks> is <generation><understanding>."
# If self_prompting is false:
#   - Personalized tokens: <sks> + shared_latent_tokens
#   - Prompt format: "<sks> is <latent_tokens>."

# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================
# Training mode: use either epoch-based (epoch > 0) or iteration-based (epoch = 0)
epoch: 0              # If > 0, train with epochs; if 0, train with iterations
iteration: 1000       # Number of iterations (used when epoch = 0)
save_every: 100       # Save checkpoint every N iterations/epochs

# Task disjoint training (only used when epoch > 0)
task_disjoin: false   # If true, uses separate tokens for generation/understanding tasks

# ==============================================================================
# DATA LOADING CONFIGURATION
# ==============================================================================
batch_size: 4         # Batch size for training and evaluation
num_workers: 1        # Number of workers for DataLoader
shuffle: true         # Shuffle training data
pin_memory: true

# ==============================================================================
# OPTIMIZATION CONFIGURATION
# ==============================================================================
optimizer:
  lr: 1.0e-4           # Learning rate
  betas: [0.9, 0.999]  # Adam betas
  weight_decay: 0.01   # Weight decay
  eps: 1.0e-8          # Epsilon for numerical stability
  grad_clip: 1.0       # Gradient clipping value (0 to disable)

scheduler:
  type: "StepLR"       # Scheduler type: "StepLR" or null
  step_size: 100       # Step size for StepLR
  gamma: 0.1           # Gamma for StepLR

# ==============================================================================
# EVALUATION CONFIGURATION
# ==============================================================================
eval:
  recognition: true                     # Enable recognition evaluation
  recognition_path_train: "data/evaluation/recognition/train"
  recognition_path_test: "data/evaluation/recognition/test"
  
  clip_sim: true                        # Enable CLIP similarity evaluation
  number_fake_images: 10                # Number of fake images to generate for CLIP eval
  
eval_visualization: true                # Generate visualization images during eval

# ==============================================================================
# FINETUNING CONFIGURATION
# ==============================================================================
finetune:
  finetune: false              # Enable/disable finetuning stage
  finetune_epoch: 2            # Number of finetuning epochs (if epoch > 0)
  finetune_iteration: 500      # Number of finetuning iterations (if epoch = 0)
  # Note: Finetuning uses positive-only dataloader

# ==============================================================================
# RESUME TRAINING CONFIGURATION
# ==============================================================================
resume:
  resume: false                # Enable/disable resuming from checkpoint
  resume_iteration: "best"     # Iteration to resume from (e.g., "best", "100", "best-gen", "best-recog")
  exp_name: "previous_exp"     # Experiment name to resume from
  savedir: "./checkpoints"     # Directory where checkpoints are saved
  
  # For task_disjoin mode (mixture of two models):
  gen_exp_name: "generation_exp"      # Generation experiment name
  understand_exp_name: "understanding_exp"  # Understanding experiment name

# ==============================================================================
# TEST/INFERENCE CONFIGURATION
# ==============================================================================
test:
  num_images: 10               # Number of images to generate
  batch_size: 2                # Batch size for generation
  prompt: "A photo."           # Prompt for image generation
  save_dir: "./generated_images"  # Directory to save generated images

# ==============================================================================
# CHECKPOINT NAMING
# ==============================================================================
# Checkpoints are saved with the following naming patterns:
#   - Regular: {iteration}-token.pt, {iteration}-lmhead.pt
#   - Finetuned: {iteration}-token-ft.pt, {iteration}-lmhead-ft.pt
#   - Best overall: best-token.pt, best-lmhead.pt
#   - Best generation: best-gen-token.pt, best-gen-lmhead.pt
#   - Best recognition: best-recog-token.pt, best-recog-lmhead.pt
#   - Whole model: {iteration}-model.pt

# ==============================================================================
# MIXED PRECISION & HARDWARE
# ==============================================================================
dtype: "bfloat16"    # Model dtype: "float32", "float16", or "bfloat16"
device: "cuda"
seed: 42

```


### Download data
```
# Mini ChameLeon
git lfs install
git clone git@huggingface.co:datasets/thaoshibe/Mini-YoChameleon-Data

# for YoLLaVA data
git lfs install
git clone https://huggingface.co/datasets/thaoshibe/YoLLaVA
```