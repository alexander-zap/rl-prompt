import os
import hydra
from omegaconf import DictConfig, OmegaConf

from rlprompt.trainers import TrainerConfig, make_trainer
from rlprompt.modules import SQLModuleConfig, make_sql_module
from rlprompt.models import (LMAdaptorModelConfig, SinglePromptModelConfig,
                             make_lm_adaptor_model, make_single_prompt_model)
from rlprompt.utils.utils import (colorful_print, compose_hydra_config_store,
                                  get_hydra_output_dir)
from helpers import (PromptedClassificationRewardConfig,
                     make_prompted_classification_reward,
                     PromptedClassificationDataset)

# Compose default config
config_list = [PromptedClassificationRewardConfig, LMAdaptorModelConfig,
               SinglePromptModelConfig, SQLModuleConfig, TrainerConfig]
cs = compose_hydra_config_store('base_fsc', config_list)


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    output_dir = get_hydra_output_dir()

    #########################################################################################
    # TODO: Add your task and data here
    train_dataset = PromptedClassificationDataset(
        ["Alexander", "Daniela", "Dieter", "Corinna", "Marius", "Michelle", "Rudi", "Ursula",
         "Siegfried", "Tina"],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    )
    val_dataset = PromptedClassificationDataset(["Michael", "Dora"], [1, 0])
    num_classes = 2
    # 0 == female name; 1 == male name
    verbalizers = ["female", "male"]
    # NOTE: You can even include a self-written prompt before the {prompt} to kick-start performance
    template = "<mask> {prompt} {sentence_1}"
    #########################################################################################

    print('Train Size:', len(train_dataset))
    print('Examples:', train_dataset[:5])
    print('Val Size', len(val_dataset))
    print('Examples:', val_dataset[:5])

    policy_model = make_lm_adaptor_model(config)
    prompt_model = make_single_prompt_model(policy_model, config)
    reward = make_prompted_classification_reward(num_classes, verbalizers,
                                                 template, config)
    algo_module = make_sql_module(prompt_model, reward, config)

    # Hack for few-shot classification - Each batch contains all examples
    config.train_batch_size = len(train_dataset)
    config.eval_batch_size = len(val_dataset)
    config.save_dir = os.path.join(output_dir, config.save_dir)
    trainer = make_trainer(algo_module, train_dataset, val_dataset, config)
    trainer.train(config=config, report_to_wandb=False)


if __name__ == "__main__":
    main()
