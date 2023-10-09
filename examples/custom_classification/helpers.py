from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import Optional, List

from reward import PromptedClassificationReward


class PromptedClassificationDataset(Dataset):
    def __init__(
            self,
            source_texts: List[str],
            class_labels: List[str]
    ):
        assert len(source_texts) == len(class_labels)
        self.source_texts = source_texts
        self.class_labels = class_labels

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        item = {'source_texts': self.source_texts[idx],
                'target_labels': self.class_labels[idx]}
        return item


def make_prompted_classification_reward(
        num_classes: int,
        verbalizers: List[str],
        template: Optional[str],
        config: "DictConfig") -> PromptedClassificationReward:
    return PromptedClassificationReward(config.task_lm, config.is_mask_lm,
                                        config.compute_zscore,
                                        num_classes, verbalizers, template)


@dataclass
class PromptedClassificationRewardConfig:
    task_lm: str = 'distilroberta-base'
    is_mask_lm: Optional[bool] = None
    compute_zscore: bool = True
