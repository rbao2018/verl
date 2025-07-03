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
import re
import gc
import time
import requests

from typing import Union, List, Optional, Dict
from functools import partial

# Constants
SUBSTITUTIONS = [
    ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""),
    (r"\ ", ""), (" ", ""), ("mbox", "text"),
    (",\\text{and}", ","), ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "hours",
    "km", "units", "\\ldots", "sue", "points", "feet", "minutes",
    "digits", "cents", "degrees", "cm", "gm", "pounds", "meters",
    "meals", "edges", "students", "childrentickets", "multiples",
    "\\text{s}", "\\text{.}", "\\text{\ns}", "\\text{}^2",
    "\\text{}^3", "\\text{\n}", "\\text{}", r"\mathrm{th}",
    r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]

def format_reward_function(predict_str: str) -> float:
    """
    Validate whether the input string strictly matches the required format:
    <think> ... </think> ... <answer> ... boxed{} ... </answer>
    
    Args:
        predict_str (str): The input string to validate.
    
    Returns:
        float: 1.0 if the input string matches the format, otherwise -1.0.
    """
    # Define the strict pattern
    pattern = re.compile(
        r'^<think>.*?</think>.*?<answer>.*?boxed\{.*?\}.*?</answer>$',
        re.DOTALL
    )
    
    # Check if the input string matches the pattern
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else -1.0


def extract_boxed_content(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string."""
    idx = string.rfind("boxed{")
    if idx < 0:
        return None

    open_braces = 0
    for i, char in enumerate(string[idx:], idx):
        if char == '{':
            open_braces += 1
        elif char == '}':
            open_braces -= 1
            if open_braces == 0:
                return string[idx:i + 1]
    return None

def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string."""
    prefix = "boxed{"
    if not s.startswith(prefix) or not s.endswith("}"):
        raise ValueError(f"Invalid boxed format: {s}")
    return s[len(prefix):-1]

def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Normalize LaTeX expressions
    patterns = [
        (r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$"),
        (r"(\\text\{)(.*?)(\})", "\\2"),
        (r"(\\textbf\{)(.*?)(\})", "\\2"),
        (r"(\\overline\{)(.*?)(\})", "\\2"),
        (r"(\\boxed\{)(.*)(\})", "\\2"),
        (r"(frac)([^{])(.)", "frac{\\2}{\\3}"),
        (r"(sqrt)([^{])", "sqrt{\\2}")
    ]
    
    for pattern, replacement in patterns:
        final_answer = re.sub(pattern, replacement, final_answer)

    final_answer = final_answer.replace("$", "")
    return final_answer.replace(",", "") if final_answer.replace(",", "").isdigit() else final_answer.strip()

def verify_minerva(solution_str: str, gt: str, gt_need_extract: bool = False) -> tuple[bool, str]:
    """Verify solution using Minerva criteria."""
    match = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)

    if gt_need_extract:
        boxed_gt = extract_boxed_content(gt)
        gt = normalize_final_answer(remove_boxed(boxed_gt)) if boxed_gt else gt
    else:
        gt = normalize_final_answer(gt)

    return (pred == gt), pred

def verify_strict_box(pred: str, gt: str) -> tuple[int, Optional[str]]:
    """Verify prediction using strict box criteria."""
    # get the last 300 tokens of the prediction
    pred = pred[-300:]

    # extract the boxed content from the prediction
    boxed_pred = extract_boxed_content(pred)

    # remove the boxed content from the prediction
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred else '[INVALID]'

    return (1 if extracted_pred == gt else -1), extracted_pred

def verify(solution_str: str, answer: str, strict_box_verify: bool = False) -> tuple[bool, str]:
    """Verify if the solution is correct."""
    if strict_box_verify:
        correct, pred = verify_strict_box(solution_str, answer)
        return correct == 1, pred

    correct, pred = verify_minerva(solution_str, answer)
    return correct, pred

def request_api_wrapper(data: dict, url: str = "http://localhost:11111/get_reward", result_key: str = "reward", max_retries: int = 2) -> float:
    """Make API request with retry logic."""
    headers = {"Content-Type": "application/json"}
    
    for _ in range(max_retries):
        try:
            response = requests.post(url=url, json=data, headers=headers, timeout=10)
            response.raise_for_status()
            result = response.json()
            if result_key not in result:
                raise KeyError(f"{result_key} not in response")
            return result[result_key]
        except Exception as e:
            print(f"API request error: {e}")
            time.sleep(3)
    return -1.0

def bailing_general_reward_func(prompt: str, response: str, label: str, strict_box_verify: bool = False) -> dict:
    """Compute the reward score for a single solution."""
    format_reward = format_reward_function(response)
    response = response[-300:]  # Limit solution length

    correct, pred = verify(response, label, strict_box_verify)

    if not correct:
        boxed_pred = extract_boxed_content(response)
        if not boxed_pred:
            correct, pred = False, "[INVALID]"
        else:
            pred = normalize_final_answer(remove_boxed(boxed_pred))
            POD_NAME = os.environ.get('POD_NAME', 'localhost')
            if "master" in POD_NAME:
                WORKER_POD_NAME = POD_NAME.replace("master", "worker")
                WORKER_0_POD_NAME = WORKER_POD_NAME.rsplit("-", 1)[0] + "-0"
                WORKER_1_POD_NAME = WORKER_POD_NAME.rsplit("-", 1)[0] + "-1"
            else:
                WORKER_0_POD_NAME, WORKER_1_POD_NAME = "localhost", "localhost"
            math_metric_1 = request_api_wrapper({"predictions": pred, "answers": label}, url=f"http://{WORKER_0_POD_NAME}:11111/get_reward")
            if math_metric_1 > 0.5:
                correct = True
            # else:
            #     math_metric_2 = request_api_wrapper({"predictions": pred, "answers": label}, url=f"http://{WORKER_1_POD_NAME}:22222/get_reward")
            #     if math_metric_2 > 0.5:
            #         correct = True

    acc_reward = 1.0 if correct else 0.0
    reward = 0.9 * acc_reward + 0.1 * format_reward

    return {
        "final_reward": reward,
        "math_verify_reward": acc_reward,
        "generative_reward": format_reward,
        "ground_truth": label,
        "prediction": pred,
    }
