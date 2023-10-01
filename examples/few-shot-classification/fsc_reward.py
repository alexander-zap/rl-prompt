import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, GPT2LMHeadModel
from typing import List, Dict, Optional, Tuple, Union, Any
from rlprompt.rewards.base_reward import BaseReward, SUPPORTED_LEFT_TO_RIGHT_LMS, SUPPORTED_MASK_LMS


class PromptedClassificationReward(BaseReward):
    def __init__(
            self,
            task_lm: str,
            is_mask_lm: Optional[bool],
            compute_zscore: bool,
            incorrect_coeff: float,  # lambda_1 in paper
            correct_coeff: float,  # lambda_2 in paper
            num_classes: int,
            verbalizers: List[str],
            template: Optional[str]
    ):

        def load_default_template() -> str:
            if self.is_mask_lm:
                mask_token = self._tokenizer.mask_token
                template = f"{{sentence_1}} {{prompt}} {mask_token} ."
            else:
                # Template for left-to-right LMs like GPT-2
                template = "{sentence_1} {prompt}"
            return template

        if is_mask_lm is None:
            # If False, then treat as left-to-right LM
            self.is_mask_lm = True if 'bert' in task_lm else False
        else:
            self.is_mask_lm = is_mask_lm

        if self.is_mask_lm:
            assert task_lm in SUPPORTED_MASK_LMS
        else:
            assert task_lm in SUPPORTED_LEFT_TO_RIGHT_LMS

        print('Task LM:', task_lm)

        super().__init__(compute_zscore)

        # Loading generator model
        if self.is_mask_lm:
            self._tokenizer = AutoTokenizer.from_pretrained(task_lm)
            self._generator = (AutoModelForMaskedLM
                               .from_pretrained(task_lm)
                               .to(self.device))
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(
                task_lm, pad_token='<|endoftext|>')
            self._generator = (GPT2LMHeadModel
                               .from_pretrained(task_lm)
                               .to(self.device))
            self._generator.config.pad_token_id = self._tokenizer.pad_token_id

        self.incorrect_coeff = incorrect_coeff
        self.correct_coeff = correct_coeff
        self.num_classes = num_classes
        self.verbalizers = verbalizers
        print('Verbalizers:', self.verbalizers)
        self.verbalizer_ids = [self._tokenizer.convert_tokens_to_ids(v)
                               for v in self.verbalizers]
        if template is None:
            self.template = load_default_template()  # prompt templates
        else:
            self.template = template

    def forward(
            self,
            source_texts: List[str],
            target_labels: List[int],
            output_tokens: List[List[str]],
            to_tensor: bool,
            mode: str
    ) -> Tuple[Union[List[float], torch.Tensor], Dict[str, Any]]:

        prompt_strings = self.get_prompt_strings_from_output_tokens(output_tokens)

        rewards = self.compute_rewards(prompt_strings=prompt_strings, source_texts=source_texts,
                                       target_labels=target_labels)

        self.rewards_per_batch.extend(rewards)

        rewards_tensor = torch.stack(self.rewards_per_batch)

        # z-score normalization (2nd stage)
        if mode == 'train' and self.compute_zscore:
            # 'z' because not source strings
            rewards_tensor = self.compute_reward_zscores(rewards_tensor=rewards_tensor,
                                                         input_texts=['z'])

        elif mode == 'infer':  # Optional: Predict Val Prompts
            score = rewards_tensor.mean().item()
            print('Our Prompt:')
            print(prompt_strings, score)

        return rewards_tensor, dict()

    def compute_rewards(self, prompt_strings, source_texts, target_labels):
        def _format_prompts(
                source_strs: List[str],
                prompt_strs: List[str],
        ) -> List[str]:
            return [self.template.format(sentence_1=s_1, prompt=p)
                    for s_1, p in zip(source_strs, prompt_strs)]

        @torch.no_grad()
        def _get_logits(texts: List[str]) -> torch.Tensor:
            # Adapted from
            # https://huggingface.co/docs/transformers/v4.21.1/en/task_summary#masked-language-modeling
            def _get_mask_token_index(input_ids: torch.Tensor) -> torch.Tensor:
                mask_token_index = torch.where(
                    input_ids == self._tokenizer.mask_token_id)[1]
                return mask_token_index

            # for MLM, add mask token
            batch_size = len(texts)
            encoded_inputs = self._tokenizer(texts, padding='longest',
                                             truncation=True, return_tensors="pt",
                                             add_special_tokens=True)

            if self.is_mask_lm:
                token_logits = self._generator(**encoded_inputs.to(self.device)).logits
                mask_token_indices = _get_mask_token_index(encoded_inputs['input_ids'])
                out_logits = token_logits[range(batch_size), mask_token_indices, :]
            else:
                token_logits = self._generator(**encoded_inputs.to(self.device)).logits
                input_lengths = encoded_inputs['attention_mask'].sum(dim=1)
                out_logits = token_logits[range(batch_size), input_lengths - 1, :]

            return out_logits

        rewards = []
        for _, prompt in enumerate(prompt_strings):
            # Compute LM logits
            current_prompts = [prompt for _ in source_texts]
            formatted_templates = _format_prompts(source_texts, current_prompts)
            all_logits = _get_logits(formatted_templates)
            # [batch_size, vocab_size]
            class_probs = torch.softmax(all_logits[:, self.verbalizer_ids], -1)
            # [batch_size, num_classes]

            batch_size = len(source_texts)
            # Get label and maximum not-label probabilities
            label_probs = class_probs[range(batch_size), target_labels]
            # [batch_size, 1]
            not_label_probs = torch.where(
                class_probs == label_probs.unsqueeze(1),
                torch.Tensor([-1]).to(self.device), class_probs)
            # [batch_size, num_classes]
            max_not_label_probs, _ = torch.max(not_label_probs, -1)
            # [batch_size, 1]

            # Compute piecewise gap reward
            gap = (label_probs - max_not_label_probs)
            correct = (gap > 0).long()

            # Log quantities such as accuracy and class-wise reward
            acc = correct.float().mean()

            gap_rewards = gap * (self.correct_coeff * correct + self.incorrect_coeff * (1 - correct))
            reward = gap_rewards.mean().detach()

            # keep track of rewards for z-score normalization
            self.input_rewards_per_batch['z'].append(reward)

            # Print examples
            print_strs = [self._counter, '|', prompt, '\n']
            for c in range(self.num_classes):
                class_example_idx = np.where(np.array(target_labels) == c)[0][0]
                class_example = formatted_templates[class_example_idx]
                class_example_probs = class_probs[class_example_idx, :].tolist()
                class_example_probs = [round(prob, 2) for prob in class_example_probs]
                print_strs += ['Class', c, 'Example:',
                               class_example, '|',
                               'Probs:', class_example_probs, '\n']
            print_strs += ['Accuracy:', acc.item(), '|',
                           'Reward:', round(reward.item(), 2)]
            print(*print_strs)

        return rewards
