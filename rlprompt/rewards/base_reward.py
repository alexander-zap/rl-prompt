from typing import Tuple, Union, List, Dict, Any
import torch
import numpy as np


class BaseReward:
    _tokenizer: Any
    _generator: Any

    def __init__(self, compute_zscore, *args, **kwargs):
        self._counter = 0
        self.compute_zscore = compute_zscore
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")

    def __call__(self, to_tensor: bool, mode: str, *args, **kwargs):
        assert mode in ["train", "infer"]

        if mode == "train":
            self._counter += 1

        rewards_tensor, rewards_log = self.forward(to_tensor=to_tensor,
                                                   mode=mode,
                                                   *args, **kwargs)

        if to_tensor is True:
            return rewards_tensor, rewards_log
        else:
            return rewards_tensor.tolist(), rewards_log

    def forward(
            self,
            source_texts: List[str],
            target_labels: List[str],
            output_tokens: List[List[str]],
            to_tensor: bool,
            mode: str
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """

        Args:
            source_texts ():
            target_labels ():
            output_tokens ():
            to_tensor ():
            mode ():

        Returns:
            rewards_tensor (torch.Tensor):
            rewards_log (Dict[str, Any]):

        """
        raise NotImplementedError

    def compute_reward_zscores(
            self,
            rewards_tensor: torch.Tensor,
            input_texts: List[str],
            input_rewards: Dict[str, List[float]],
            eps: float = 1e-4
    ) -> torch.Tensor:
        input_reward_means = {k: np.mean(v) for k, v in input_rewards.items()}
        input_reward_stds = {k: np.std(v) for k, v in input_rewards.items()}
        idx_means = torch.tensor([input_reward_means[s] for s in input_texts])
        idx_stds = torch.tensor([input_reward_stds[s] for s in input_texts])
        # print(idx_means)
        # print(idx_stds)
        return (rewards_tensor - idx_means.float()) / (idx_stds.float() + eps)

    def _convert_tokens_to_string(self, tokens: List[List[str]]) -> List[str]:
        return [self._tokenizer.convert_tokens_to_string(s)
                for s in tokens]
