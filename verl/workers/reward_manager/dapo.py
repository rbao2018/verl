# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import time

import torch

from verl import DataProto
from verl.utils.reward_score.math_dapo_pebble import compute_score as math_dapo_pebble_compute_score
from verl.workers.reward_manager import register


@register("dapo")
class DAPORewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = math_dapo_pebble_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def verify(self, data):
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_strs, ground_truths, data_sources, extras = [], [], [], []
        responses_ids = []
        
        start_time = time.time()  # 记录开始时间
        print("detokenizer start now!", flush=True)

        for i in range(len(data)):
            data_item = data[i]
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            responses_ids.append(valid_response_ids)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_strs.append(response_str)
            # Get the ground truth and data source
            ground_truths.append(data_item.non_tensor_batch["reward_model"].get("ground_truth", None))
            data_sources.append(data_item.non_tensor_batch[self.reward_fn_key])
            extras.append(data_item.non_tensor_batch.get("extra_info", None))
        # If the compute_score function is not provided, use the default one

        total_time = time.time() - start_time
        print(f"Total detokenizer time taken: {total_time:.2f} seconds", flush=True)

        results = self.compute_score(
            data_sources=data_sources,
            solution_strs=responses_strs,
            ground_truths=ground_truths,
            extra_infos=extras
        )

        return results, responses_ids, ground_truths, data_sources

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        
        # print(f"data.non_tensor_batch.keys() are :{data.non_tensor_batch.keys()}", flush=True)

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        valid_response_lengths = data.batch["attention_mask"][:, prompt_ids.shape[-1]:].sum(dim=-1)

        already_print_data_sources = {}

        if "__final_reward__" in data.non_tensor_batch.keys():
            results, responses_ids, ground_truths, data_sources = [], [], [], []
            for i in range(len(data)):
                data_item = data[i]
                # results.append(data_item.non_tensor_batch["__final_reward__"])
                results.append(
                    {
                        "score": data_item.non_tensor_batch["__final_reward__"],
                        "pred": data_item.non_tensor_batch["__prediction__"]
                    }
                )
                responses_ids.append(data.batch["responses"][i][:valid_response_lengths[i]])
                ground_truths.append(data_item.non_tensor_batch["__ground_truth__"])
                data_sources.append(data_item.non_tensor_batch[self.reward_fn_key])
        else:
            results, responses_ids, ground_truths, data_sources = self.verify(data)
        
        for i in range(len(data)):
            # Get the data for this instance
            valid_response_length = valid_response_lengths[i].item()
            # Get the result (dict or float) and the data source
            result, data_source = results[i], data_sources[i]
            # Get the response string, output string, and ground truth
            response_id, ground_truth = responses_ids[i], ground_truths[i]

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
            
            overlong_reward = 0
            reward = score
        
            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = torch.tensor(reward, dtype=torch.float32)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                prompt_str = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
                print("[data_source]", data_source, flush=True)
                print("[prompt]", prompt_str, flush=True)
                print("[response]", self.tokenizer.decode(response_id, skip_special_tokens=False), flush=True)
                print("[ground_truth]", ground_truth, flush=True)
                print("[valid_response_length]", valid_response_length, flush=True)
                print("[overlong_reward]", overlong_reward, flush=True)
                
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value, flush=True)
                else:
                    print("[score]", score, flush=True)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

