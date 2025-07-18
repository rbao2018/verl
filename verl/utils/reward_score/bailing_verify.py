# Copyright 2025 Ant Group. and/or its affiliates
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

import os
import json
import gc
import time
import requests

from typing import Union, List, Optional, Dict
from functools import partial
from pebble import ProcessPool


def get_verifier_rewards(examples):
    responses = [example['response'] for example in examples]
    labels = [example['label'] for example in examples]

    request_data = {
        "verifier": [label.get("verifier", None) for label in labels],
        "pred": responses,
        "gold": [label.get("gold", None) for label in labels],
        "template_answer": [label.get("template_answer", "") for label in labels]
    }

    data = {
        "invokeSource": "antopt",
        "staffNo": "377381",
        "async": False,
        "modelName": "rule_base_instruct_follow",
        "requestData": request_data,
        "apiId": "QywxxoZDoDddxahzetLG"
    }

    url = "https://antoptplatform.alipay.com/api/v1/ds/service"
    headers = {"Content-Type": "application/json"}

    for _ in range(3):  # Retry up to 3 times
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
            is_correct_list = [item["pass"] for item in response.json()['data']['reward_score']]
            rewards = [1.0 if is_correct else 0.0 for is_correct in is_correct_list]
            break
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            time.sleep(1)
    
    return rewards


def process_batch(batch):
    examples = [{"pred": sol, "label": json.loads(gt)} for sol, gt in batch]
    return get_verifier_rewards(examples)

def compute_score(solution_strs: Union[str, List[str]], ground_truths: Union[str, List[str]], timeout_score=0) -> Union[float, List[float]]:
    """Compute reward scores for solutions using parallel processing."""

    # Handle single task case
    if isinstance(solution_strs, str) and isinstance(ground_truths, str):
        examples = [{"pred": solution_strs, "label": json.loads(ground_truths)}]
        result = get_verifier_rewards(examples)[0]
        return result

    # Handle multiple tasks case
    if not isinstance(solution_strs, list) or not isinstance(ground_truths, list):
        raise ValueError("Both solution_strs and ground_truths must be either strings or lists")
    
    if len(solution_strs) != len(ground_truths):
        raise ValueError("solution_strs and ground_truths must have equal length")

    # Handle empty list case
    if len(solution_strs) == 0:
        return []

    # Split tasks into batches
    tasks = list(zip(solution_strs, ground_truths))
    batch_size = min(1024, len(tasks))

    batches = [tasks[i:i+batch_size] for i in range(0, len(tasks), batch_size)]

    results = []

    with ProcessPool() as pool:
        futures = [pool.schedule(process_batch, args=(batch,)) for batch in batches]
        
        # Wait for all tasks to complete
        for future, batch in zip(futures, batches):
            try:
                batch_results = future.result()
                results.extend(batch_results)
            except Exception as e:
                print(f"Error processing batch: {e}")
                results.extend([timeout_score] * len(batch))

    return results


# def compute_score(solution_str: str, ground_truth: str, extra_info: Optional[Dict] = None) -> float:
#     """Compute the score for a given solution based on the bailing verifier.

#     Args:
#         solution_str (str): The solution string to be evaluated.
#         ground_truth (str): The ground truth answer for comparison.
#         extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

#     Returns:
#         float: The computed score as a floating point number.
#     """
#     if extra_info is None:
#         extra_info = {}

#     example = {
#         'pred': solution_str,
#         'label': json.loads(ground_truth)
#     }

#     return get_verifier_reward(example)
