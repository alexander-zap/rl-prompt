import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, GPT2LMHeadModel
from typing import List, Dict, Optional, Tuple, Union, Any
from rlprompt.rewards.base_reward import BaseReward


class PromptedClassificationReward(BaseReward):
    def __init__(
            self,
            task_lm: str,
            is_mask_lm: Optional[bool],
            compute_zscore: bool,
            num_classes: int,
            verbalizers: List[str],
            template: Optional[str]
    ):

        if is_mask_lm is None:
            # If False, then treat as left-to-right LM
            self.is_mask_lm = True if 'bert' in task_lm else False
        else:
            self.is_mask_lm = is_mask_lm

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

        self.num_classes = num_classes
        self.verbalizers = verbalizers
        print('Verbalizers:', self.verbalizers)
        self.verbalizer_ids = [self._tokenizer.convert_tokens_to_ids(v)
                               for v in self.verbalizers]

        self.template = template

    def compute_rewards(
            self,
            prompt_strings: List[str],
            source_texts: List[str],
            target_labels: List[int],
            mode: str
    ) -> Tuple[Union[List[float], torch.Tensor], Dict[str, Any]]:

        def compute_reward(prompt, source, label):

            def format_prompt(
                    source_str: str,
                    prompt_str: str,
            ) -> str:
                return self.template.format(sentence_1=source_str, prompt=prompt_str)

            @torch.no_grad()
            def get_logits(text: str) -> torch.Tensor:
                # Adapted from
                # https://huggingface.co/docs/transformers/v4.21.1/en/task_summary#masked-language-modeling
                def _get_mask_token_index(input_ids: torch.Tensor) -> torch.Tensor:
                    mask_token_index = torch.where(
                        input_ids == self._tokenizer.mask_token_id)[1]
                    return mask_token_index

                # for MLM, add mask token
                encoded_inputs = self._tokenizer([text], padding='longest',
                                                 truncation=True, return_tensors="pt",
                                                 add_special_tokens=True)
                if self.is_mask_lm:
                    token_logits = self._generator(**encoded_inputs.to(self.device)).logits
                    mask_token_indices = _get_mask_token_index(encoded_inputs['input_ids'])
                    out_logits = token_logits[0, mask_token_indices, :]
                else:
                    token_logits = self._generator(**encoded_inputs.to(self.device)).logits
                    input_lengths = encoded_inputs['attention_mask'].sum(dim=1)
                    out_logits = token_logits[0, input_lengths - 1, :]

                return out_logits

            def log_metrics():
                # Print examples
                print_strs = [self._counter, '|', prompt_string, '\n', ]
                for c in range(self.num_classes):
                    class_example = formatted_template
                    print_strs += ['Class', c, 'Example:',
                                   class_example, '|',
                                   'Probability:', class_probs.tolist()[c], '\n']
                print_strs += ['Reward:', round(reward, 2)]
                print(*print_strs)

            # Compute LM logits
            formatted_template = format_prompt(source, prompt)
            all_logits = get_logits(formatted_template)
            # [1, vocab_size]
            class_probs = torch.softmax(all_logits[:, self.verbalizer_ids], -1)[0]
            # [1, num_classes]

            if torch.argmax(class_probs) == label:
                reward = 1.0
            else:
                reward = -1.0

            log_metrics()

            return reward

        rewards = []
        for prompt_string, source_text, target_label in zip(prompt_strings, source_texts, target_labels):
            reward = compute_reward(prompt_string, source_text, target_label)
            rewards.append(torch.tensor(reward, dtype=torch.float))

        rewards_tensor = torch.stack(rewards)

        print(f"Average Reward: {rewards_tensor.mean()}")
        return rewards_tensor, dict()
