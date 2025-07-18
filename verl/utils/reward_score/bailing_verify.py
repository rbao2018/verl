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


def get_verifier_reward(example, max_retries: int = 3) -> float:
    pred, label = example['pred'], example['label']
    
    request_data = {
        "verifier": [label.get("verifier", None)],
        "pred": [pred],
        "gold": [label.get("gold", None)],
        "template_answer": [label.get("template_answer", "")]
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

    for _ in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
            is_correct = response.json()['data']['reward_score'][0]["pass"]
            reward = 1.0 if is_correct else 0.0
            return reward
        except Exception as e:
            print(f"Request failed: {e}")
            time.sleep(1)

    return 0.0


def compute_score(solution_str: str, ground_truth: str, extra_info: Optional[Dict] = None) -> float:
    """Compute the score for a given solution based on the bailing verifier.

    Args:
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number.
    """
    if extra_info is None:
        extra_info = {}

    example = {
        'pred': solution_str,
        'label': json.loads(ground_truth)
    }

    return get_verifier_reward(example)
