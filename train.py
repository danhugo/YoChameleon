import argparse
import warnings

import yaml
from utils.helpers import get_dataloader_iter, get_eval_data
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

    # call training loop
    if config.model_id == ChameleonModelName.LENOY_ANOLE_7B_V01:
        trainer = YoChameleonTrainer(config)
    else:
        raise ValueError(f"Model ID {config.model_id} is not supported yet~")

    personalized_prompt = trainer.get_personalized_prompt()
    logger.info(f"Personalized prompt: {personalized_prompt}")
    
    train_dataloader = get_dataloader_iter(
        config=config,
        processor=trainer.processor,
        personalized_prompt=personalized_prompt
        )

    trainer.resume_training()
    trainer.configure_model() # this step will set up optimization

    if config.self_prompting:
        understanding_prompt = trainer.get_understanding_prompt()
    else:
        understanding_prompt = None
    _, recognition_data_train = get_eval_data(
        config,
        trainer.processor,
        image_folder=config.eval.recognition_path_train,
        personalized_prompt=personalized_prompt,
        understanding_prompt=understanding_prompt
    )
    _, recognition_data_test = get_eval_data(
        config,
        trainer.processor,
        image_folder=config.eval.recognition_path_test,
        personalized_prompt=personalized_prompt,
        understanding_prompt=understanding_prompt
    )
    if config.epoch > 0: # Now only train with epoch
        config.iteration = config.epoch

        trainer.train(
            train_dataloader,
            recognition_data_train,
            recognition_data_test
            )
    else:
        raise ValueError("C'mon you need to train at least one epoch.")