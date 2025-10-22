import argparse
import warnings

import yaml
from utils.helpers import get_dataloader_iter, get_eval_dataloader
from yochameleon import YoChameleonTrainer
from config.config import GeneralConfig
from utils.logging import get_logger
from config.consts import ChameleonModelName

logger = get_logger(__name__)

# from yoemu3 import YoEmu3Trainer

warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description='Your Chameleon model')
    parser.add_argument('--config', type=str, default='./config/basic.yml')
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()
    config_dict = yaml.safe_load(open(args.config, 'r'))
    config = GeneralConfig(**config_dict)

    # Load the universal config only, override sks name with actual sks_name
    # TODO: check for SKS_NAME in json_file
    if config.sks_name is not None:
        config.json_file = [x.replace('SKS_NAME', config.sks_name) for x in config.json_file]

    # call training loop
    if config.model_id == ChameleonModelName.LENOY_ANOLE_7B_V01:
        trainer = YoChameleonTrainer(config)
    # elif config.model_id == 'Emu3-community/Emu3-Gen-hf':
    #     trainer = YoEmu3Trainer(config)
    else:
        raise ValueError(f"Model ID {config.model_id} is not supported yet~")

    #TODO: dive here
    personalized_prompt = trainer.get_personalized_prompt()
    logger.info(f"Personalized prompt: {personalized_prompt}")
    
    #TODO: dive here
    train_dataloader = get_dataloader_iter(
        config,
        trainer.processor,
        personalized_prompt=personalized_prompt
        )

    trainer.resume_training()
    trainer.configure_model() # this step will set up optimization
    #TODO: dive here
    if config.self_prompting:
        understanding_prompt = trainer.get_understanding_prompt()
    else:
        understanding_prompt = None
    recognition_dataloader_train = get_eval_dataloader(
        config,
        trainer.processor,
        image_folder=config.eval['recognition_path_train'],
        personalized_prompt=personalized_prompt,
        understanding_prompt=understanding_prompt
    )
    recognition_dataloader_test = get_eval_dataloader(
        config,
        trainer.processor,
        image_folder=config.eval['recognition_path_test'],
        personalized_prompt=personalized_prompt,
        understanding_prompt=understanding_prompt
    )
    if config.epoch > 0: #If you want to train with epoch... Fine, here you go
        config.iteration = config.epoch

        trainer.train_epoch(
            train_dataloader,
            recognition_dataloader_train,
            recognition_dataloader_test
            )

        # -- Thao: Maybe we should move this to the finetuning stage for all
        if config.finetune['finetune']:
            config.finetune['finetune_iteration'] = config.finetune['finetune_epoch']
            positive_only_dataloader = get_dataloader_iter(config, trainer.processor, only_positive=True)
            trainer.finetune_epoch(positive_only_dataloader)

    else: # This support train with iteration
        print('Hello, train with iteration')
        trainer.train(train_dataloader)
        if config.finetune['finetune']:
            positive_only_dataloader = get_dataloader_iter(config, trainer.processor, only_positive=True)
            trainer.finetune(positive_only_dataloader)
    # trainer.train(train_dataloader)
