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

        super().__init__(compute_zscore=compute_zscore)

        # Loading generator model
        self._tokenizer = AutoTokenizer.from_pretrained(task_lm)
        self._generator = PromptedGenerator(task_lm, template, end_punct,
                                            pad_token, self.device,
                                            lower_outputs, control_output_length)

        self.top_k = task_top_k
        self.top_p = 1.0
        self.num_samples = num_samples
        self.num_bootstraps = num_bootstraps

        # Loading reward models
        if style_tokenizer is None:
            style_tokenizer = style_classifier
        self.selector = TextStyleTransferOutputSelector(style_classifier,
                                                        style_tokenizer,
                                                        style_batch_size,
                                                        self.device)

        # Misc. training details
        self.num_repeats = num_repeats
        self.tokens_explored = set()

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
        elif mode == "infer":
            source_strs = source_texts

        prompt_strings = self.calculcate_prompt_strings(output_tokens)

        n_reward = self.num_samples
        k_reward = self.num_bootstraps
        N = n_reward * k_reward

        rewards: List[torch.Tensor] = []
        input_rewards: Dict[str, List[float]] = defaultdict(list)
        quantities_to_log: Dict[str, List[torch.Tensor]] = defaultdict(list)
        for i, (prompt, src, label) in enumerate(zip(prompt_strings,
                                                     source_strs,
                                                     target_labels)):
            hypos = self._generator.sample_generate(prompt, src, N,
                                                    self.top_k, self.top_p)
            sum_rewards, content_scores, style_probs = \
                self.selector.compute_sample_rewards(src, hypos, label)

            # Bootstrap the max reward for k times and average
            bootstrap_max_rewards: List[float] = \
                self._boostrap_max_rewards_k_times(sum_rewards, k_reward)
            # Average boostrap max rewards as the final reward
            reward = torch.Tensor(bootstrap_max_rewards).float().mean()

            # Keep track of each input's max rewards to compute z-score
            input_rewards[src] += bootstrap_max_rewards

            # Take the max of the sub-list rewards to print as example
            max_reward = max(bootstrap_max_rewards)
            top_index = sum_rewards.index(max_reward)

            # Log relevant quantities
            # mean content
            content = torch.tensor(content_scores).float().mean()
            # mean style
            prob = torch.tensor(style_probs).float().mean()
            mean_reward = torch.tensor(sum_rewards).float().mean()
            top_content = torch.tensor(content_scores[top_index]).float()
            # top style
            top_prob = torch.tensor(style_probs[top_index]).float()


            print(self._counter, '|', output_tokens[i], '|',
                  prompt, '|', src, '|', hypos[top_index], '|',
                  'Top Content:', round(top_content.item(), 2), '|',
                  'Top Style:', round(top_prob.item(), 2), '|',
                  'Top Reward:', round(max_reward, 2), '|',
                  'Reward:', round(reward.item(), 2))
            rewards.append(reward)

        rewards_tensor = torch.stack(rewards)
        if mode == "train" and self.compute_zscore:
            rewards_tensor = self.compute_reward_zscores(rewards_tensor,
                                                         source_strs,
                                                         input_rewards)

        self.tokens_explored = \
            self.tokens_explored.union(*[set(p) for p in output_tokens])

        num_tokens_explored = torch.tensor(len(self.tokens_explored)).float()

        rewards_log = dict()

        return rewards_tensor, rewards_log

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

