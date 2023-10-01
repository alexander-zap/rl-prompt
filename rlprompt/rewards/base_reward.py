from typing import Tuple, List, Dict, Any
import torch
import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod

# Hardcoded
SUPPORTED_LEFT_TO_RIGHT_LMS = ['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
SUPPORTED_MASK_LMS = ['distilroberta-base', 'roberta-base', 'roberta-large']


class BaseReward(ABC):
    _tokenizer: Any
    _generator: Any

    def __init__(self, compute_zscore, *args, **kwargs):
        self._counter = 0
        self.compute_zscore = compute_zscore
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokens_explored = set()

        self.rewards_per_batch: List[torch.Tensor] = []
        self.input_rewards_per_batch: Dict[str, List[float]] = defaultdict(list)

    def __call__(self, output_tokens: List[List[str]], to_tensor: bool, mode: str,
                 *args, **kwargs):
        assert mode in ["train", "infer"]

        if mode == "train":
            self._counter += 1

        self.tokens_explored = self.tokens_explored.union(*[set(p) for p in output_tokens])

        self.rewards_per_batch = []
        self.input_rewards_per_batch = defaultdict(list)
        rewards_tensor, rewards_log = self.forward(output_tokens=output_tokens,
                                                   to_tensor=to_tensor,
                                                   mode=mode,
                                                   *args, **kwargs)

        if to_tensor is True:
            return rewards_tensor, rewards_log
        else:
            return rewards_tensor.tolist(), rewards_log

    @abstractmethod
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
            eps: float = 1e-4
    ) -> torch.Tensor:
        input_reward_means = {k: np.mean(v) for k, v in self.input_rewards_per_batch.items()}
        input_reward_stds = {k: np.std(v) for k, v in self.input_rewards_per_batch.items()}
        idx_means = torch.tensor([input_reward_means[s] for s in input_texts])
        idx_stds = torch.tensor([input_reward_stds[s] for s in input_texts])
        return (rewards_tensor - idx_means.float()) / (idx_stds.float() + eps)

    def get_prompt_strings_from_output_tokens(self, output_tokens):
        """
        Process prompts and verbalizer indices

        Args:
            output_tokens ():

        Returns:
            prompt_strings

        """

        def _convert_tokens_to_string(tokens: List[List[str]]) -> List[str]:
            return [self._tokenizer.convert_tokens_to_string(s)
                    for s in tokens]

        prompt_tokens = output_tokens
        prompt_strings = _convert_tokens_to_string(prompt_tokens)
        return prompt_strings

    @abstractmethod
    def compute_rewards(self, prompts, source_strings, labels):
        """

        Args:
            prompts ():
            source_strings ():
            labels ():

        Returns:
            rewards (List[Tensor]): List of rewards with same length as input.
                NOTE: Each Tensor-object is 1-dimensional.

        """
        raise NotImplementedError
