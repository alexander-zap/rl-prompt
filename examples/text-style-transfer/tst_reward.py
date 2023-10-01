import torch
import itertools
from typing import List, Tuple, Union, Dict, Any, Optional
from transformers import AutoTokenizer
from collections import defaultdict
from tst_modules import PromptedGenerator, TextStyleTransferOutputSelector
from rlprompt.rewards.base_reward import BaseReward, SUPPORTED_LEFT_TO_RIGHT_LMS


class PromptedTextStyleTransferReward(BaseReward):
    def __init__(
            self,
            task_lm: str,
            task_top_k: int,  # Top-k sampling for text generation
            style_classifier: str,
            style_tokenizer: Optional[str],
            style_batch_size: int,
            pad_token: str,
            num_repeats: int,  # Num of repetitions for each example
            num_samples: int,  # Num of samples from which to take the output
            num_bootstraps: int,  # Num of bootstraps to reduce reward randomness
            compute_zscore: bool,  # Whether to compute z-score of rewards
            lower_outputs: bool,  # Whether to convert all outputs to lower case
            control_output_length: bool,  # Control output length for speedup
            template: str,  # Template for prompt generation
            end_punct: str,  # End punctuation to cut off after generation
    ):

        assert task_lm in SUPPORTED_LEFT_TO_RIGHT_LMS

        print('Task LM:', task_lm)

        super().__init__(compute_zscore)

        # Loading generator model
        self._tokenizer = AutoTokenizer.from_pretrained(task_lm)
        self._generator = PromptedGenerator(task_lm, template, end_punct,
                                            pad_token, self.device,
                                            lower_outputs, control_output_length)

        # Loading reward models
        if style_tokenizer is None:
            style_tokenizer = style_classifier
        self.selector = TextStyleTransferOutputSelector(style_classifier,
                                                        style_tokenizer,
                                                        style_batch_size,
                                                        self.device)

        self.top_k = task_top_k
        self.top_p = 1.0
        self.num_samples = num_samples
        self.num_bootstraps = num_bootstraps

        # Misc. training details
        self.num_repeats = num_repeats

    def forward(
            self,
            source_texts: List[str],
            target_labels: List[str],
            output_tokens: List[List[str]],
            to_tensor: bool,
            mode: str
    ) -> Tuple[Union[List[float], torch.Tensor], Dict[str, Any]]:

        if mode == 'train':
            source_strs = self._repeat_texts(source_texts)
            target_labels = self._repeat_texts(target_labels)
        else:  # mode == "infer":
            source_strs = source_texts

        prompt_strings = self.get_prompt_strings_from_output_tokens(output_tokens)

        for i, (prompt, src, label) in enumerate(zip(prompt_strings,
                                                     source_strs,
                                                     target_labels)):
            reward = self.compute_reward(prompt, src, label)
            self.rewards_per_batch.append(reward)

        rewards_tensor = torch.stack(self.rewards_per_batch)

        if mode == "train" and self.compute_zscore:
            rewards_tensor = self.compute_reward_zscores(rewards_tensor,
                                                         source_strs)

        return rewards_tensor, dict()

    def _boostrap_max_rewards_k_times(
            self,
            rewards: List[float],
            k: int
    ) -> List[float]:
        # Segment list rewards into k equal sub-lists
        l = len(rewards)
        assert l % k == 0, f'l={l}, k={k}'
        segmented_rewards = [rewards[i * l // k:(i + 1) * l // k]
                             for i in range(k)]  # [k, l/k]
        # We use different rewards for each bootstrap for now
        bootstrap_rewards = segmented_rewards

        # For each sub-list, take the max as the sub-reward
        values, indices = (torch.tensor(bootstrap_rewards)
                           .float().max(axis=1))
        # Take numbers from the original list to avoid numerical issues
        bootstrap_max_rewards = [bootstrap_rewards[i][index]
                                 for i, index in enumerate(indices)]

        return bootstrap_max_rewards

    def _repeat_texts(
            self,
            texts: List[str],
            num_repeats: Optional[int] = None
    ) -> List[str]:
        if num_repeats is None:
            num_repeats = self.num_repeats
        return list(itertools.chain(*[[s for _ in range(num_repeats)]
                                      for s in texts]))

    def compute_reward(self, prompt, src, label):

        def log_metrics():
            # Take the max of the sub-list rewards to print as example
            mean_reward = torch.tensor(sum_rewards).float().mean()
            max_reward = max(bootstrap_max_rewards)
            top_index = sum_rewards.index(max_reward)
            mean_content = torch.tensor(content_scores).float().mean()
            top_content = torch.tensor(content_scores[top_index]).float()
            mean_style = torch.tensor(style_probs).float().mean()
            top_style = torch.tensor(style_probs[top_index]).float()
            num_tokens_explored = torch.tensor(len(self.tokens_explored)).float()

            print(self._counter, '|',
                  prompt, '|',
                  src, '|',
                  hypos[top_index], '|',
                  'Mean Content:', round(mean_content.item(), 2), '|',
                  'Mean Style:', round(mean_style.item(), 2), '|',
                  'Mean Reward:', round(mean_reward.item(), 2), '|',
                  'Top Content:', round(top_content.item(), 2), '|',
                  'Top Style:', round(top_style.item(), 2), '|',
                  'Top Reward:', round(max_reward, 2), '|',
                  'Reward:', round(reward.item(), 2), '|',
                  '# Explored Tokens', num_tokens_explored
                  )

        n_reward = self.num_samples
        k_reward = self.num_bootstraps
        num_samples = n_reward * k_reward

        hypos = self._generator.sample_generate(prompt, src, num_samples, self.top_k, self.top_p)
        sum_rewards, content_scores, style_probs = self.selector.compute_sample_rewards(src, hypos, label)

        # Bootstrap the max reward for k times and average
        bootstrap_max_rewards: List[float] = self._boostrap_max_rewards_k_times(sum_rewards, k_reward)

        # Keep track of each input's max rewards to compute z-score
        self.input_rewards_per_batch[src] += bootstrap_max_rewards

        # Average boostrap max rewards as the final reward
        reward = torch.Tensor(bootstrap_max_rewards).float().mean()

        log_metrics()

        return reward
