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

# try:
#     from math_verify.errors import TimeoutException
#     from math_verify.metric import math_metric
#     from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
# except ImportError:
#     print("To use Math-Verify, please install it first by running `pip install math-verify`.")


# def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
#     verify_func = math_metric(
#         gold_extraction_target=(LatexExtractionConfig(),),
#         pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
#     )
#     ret_score = 0.0

#     # Wrap the ground truth in \boxed{} format for verification
#     ground_truth_boxed = "\\boxed{" + ground_truth + "}"
#     try:
#         ret_score, _ = verify_func([ground_truth_boxed], [model_output])
#     except Exception:
#         pass
#     except TimeoutException:
#         ret_score = timeout_score

#     return ret_score


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
from pebble import ProcessPool

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

def request_api_wrapper(data: dict, url: str = "http://localhost:11111/get_reward", result_key: str = "reward", max_retries: int = 3) -> float:
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

def compute_single_score(solution_str: str, ground_truth: str, timeout_score: float = 0.0) -> float:
    """Compute the reward score for a single solution."""
    # format_reward = format_reward_function(solution_str)
    # solution_str = solution_str[-300:]  # Limit solution length

    # boxed_pred = extract_boxed_content(solution_str)
    # if not boxed_pred:
    #     correct, pred = False, "[INVALID]"
    # else:
    #     pred = normalize_final_answer(remove_boxed(boxed_pred))
    #     math_metric_1 = request_api_wrapper({"predictions": pred, "answers": ground_truth}, url=f"http://localhost:11111/get_reward")
    #     if math_metric_1 > 0.5:
    #         correct = True

    math_metric_1 = request_api_wrapper({"predictions": solution_str, "answers": ground_truth}, url=f"http://localhost:11111/get_reward")
    if math_metric_1 > 0.5:
        correct = True

    acc_reward = 1.0 if correct else timeout_score

    return acc_reward

def process_task(args: tuple, timeout_score: float = 0) -> float:
    """Process a single task with solution and ground truth."""
    try:
        solution, ground_truth = args
        return compute_single_score(solution, ground_truth, timeout_score)
    except Exception as e:
        return timeout_score

def compute_score(solution_strs: Union[str, List[str]], ground_truths: Union[str, List[str]], timeout_score=0) -> Union[float, List[float]]:
    """Compute reward scores for solutions using parallel processing."""

    # Handle single task case
    if isinstance(solution_strs, str) and isinstance(ground_truths, str):
        result = process_task((solution_strs, ground_truths), timeout_score)
        return result

    # Handle multiple tasks case
    if not isinstance(solution_strs, list) or not isinstance(ground_truths, list):
        raise ValueError("Both solution_strs and ground_truths must be either strings or lists")
    
    if len(solution_strs) != len(ground_truths):
        raise ValueError("solution_strs and ground_truths must have equal length")

    tasks = list(zip(solution_strs, ground_truths))
    results = []

    with ProcessPool(max_workers=min(128, os.cpu_count() - 32)) as pool:
        process_func = partial(process_task, timeout_score=timeout_score)

        futures = [pool.schedule(process_func, args=(task,)) for task in tasks]
        results = [future.result() for future in futures]

    return results

