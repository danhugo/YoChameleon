from pydantic import BaseModel, Field, computed_field
from typing import Literal
from config.consts import AllModelEnum, ChameleonModelName
import os

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
    gen_exp_name: str | None
    understand_exp_name: str | None


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
    vqa: bool
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
    savedir: str
    no_wandb: bool = Field(description="Turn off log to WanDB for debug")

    sks_name: str = Field(description="name of personalized object to overwrite SKS_NAME in json_file")
    @computed_field
    @property
    def json_file(self) -> list[str]:
        files = ['recognition.json', 'text_conversation.json', '1000.json']
        return [os.path.join(self.data_root, "train", self.sks_name, "json", f) for f in files]
    
    model_id: AllModelEnum = ChameleonModelName.LENOY_ANOLE_7B_V01
    data_root: str

    prefix_token: int

    different_identifier: bool
    self_prompting: bool
    
    @computed_field
    @property
    def save_location(self) -> str:
        return f'{self.savedir}/{self.exp_name}/{self.sks_name}'

    iteration: int
    epoch: int
    batch_size: int

    save_every: int

    whole_model: bool
    tokenizer_max_length: int
    eval_visualization: bool | None

    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

    special_tokens: SpecialTokenConfig
    resume: ResumeConfig | None
    finetune: FineTuneConfig | None
    eval: EvalConfig
    test: TestConfig

    class Config:
        extra = 'ignore'
