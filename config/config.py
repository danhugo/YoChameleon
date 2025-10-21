from pydantic import BaseModel, Field
from typing import Literal
from config.consts import AllModelEnum, ChameleonModelName


class OptimizerConfig(BaseModel):
    type: str
    lr: float
    betas: list[float]
    weight_decay = float
    eps: float
    grad_clip: float


class SchedulerConfig(BaseModel):
    type: str
    step_size: int
    gamma: float


class ResumeConfig(BaseModel):
    resume: Literal["yes", "no"]
    resume_iteration: int
    savedir: str
    exp_name: str


class FineTuneConfig(BaseModel):
    finetune: bool
    finetune_iteration: int
    finetune_epoch: int
    save_every: int
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig


class EvalConfig(BaseModel):
    clip_sim: bool
    number_fake_images: int
    recognition: bool
    recognition_path_train: str
    recognition_path_test: str
    vpg: bool
    vqa_path_json: str


class TestConfig(BaseModel):
    prompt: str
    token_len: int | None
    iteration: int
    save_dir: str
    batch_size: int
    num_images: int


class SpecialTokenConfig(BaseModel):
    START_OF_IMAGE_INDEX: int
    END_OF_IMAGE_INDEX: int
    END_OF_TURN: int
    PAD_INDEX: int
    SKS_TOKEN: str
    LATENT_TOKEN_START: int



class GeneralConfig(BaseModel):
    project_name: str = "YoChameleon"
    entity: str
    exp_name: str
    
    model_id: AllModelEnum = ChameleonModelName.LENOY_ANOLE_7B_V01
    data_root: str

    no_wandb: bool = Field(description="Turn off log to WanDB for debug")

    json_file: list[str] = Field(examples=[
        'data/train/SKS_NAME/json/recognition.json',
        'data/train/SKS_NAME/json/text_conversation.json',
        'data/train/SKS_NAME/json/1000E.json'
    ])

    sks_name: str = Field(description="name of personalized object to overwrite SKS_NAME in json_file")
    prefix_token: int

    different_identifier: bool
    task_disjoin: bool
    self_prompting: bool
    seperate_tasks: bool = Field(
        default=False, 
        description="the model will be trained on all tasks (recognition, gen) with same tokens, or seperate"
        )

    iteration: int
    epoch: int
    batch_size: int
    savedir: str
    save_every: int

    whole_model: bool
    tokenizer_max_length: int
    eval_visualization: bool

    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

    special_tokens: SpecialTokenConfig
    resume: ResumeConfig
    finetune: FineTuneConfig
    eval: EvalConfig
    test: TestConfig

    class Config:
        extra = 'ignore'
