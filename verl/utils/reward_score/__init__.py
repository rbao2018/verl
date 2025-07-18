# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from typing import Union, List, Dict

from verl.utils.import_utils import deprecated

def default_compute_score(data_sources: Union[str, List[str]],
                          solution_strs: Union[str, List[str]],
                          ground_truths: Union[str, List[str]],
                          extra_infos: Union[Dict, List[Dict]] = None,
                          sandbox_fusion_url: str = None,
                          concurrent_semaphore: int = None,
                          memory_limit_mb: int = None) -> Union[float, Dict, List]:
    """Compute the score for a given solution based on the data source.

    Args:
        data_sources (str): The source dataset identifier which determines the scoring method.
        solution_strs (str): The solution string to be evaluated.
        ground_truths (str): The ground truth answer for comparison.
        extra_infos (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """

    if isinstance(data_sources, str):
        data_source = data_sources
    else:
        data_source = data_sources[0]

    if data_source == "openai/gsm8k":
        from . import gsm8k
        res = gsm8k.compute_score(solution_strs, ground_truths)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        # from . import math

        # res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        from . import math_verify
        res = math_verify.compute_score(solution_strs, ground_truths)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_strs, ground_truths)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_strs, ground_truths)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_strs, ground_truths, continuous=True)
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_strs, ground_truths, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_strs, ground_truths)
    elif data_source in ["searchR1_nq", "searchR1_triviaqa", "searchR1_popqa", "searchR1_hotpotqa", "searchR1_2wikimultihopqa", "searchR1_musique", "searchR1_bamboogle"]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_strs, ground_truths)

    elif data_source in ["bailing_verify"]:
        from . import bailing_verify
        res = bailing_verify.compute_score(solution_strs, ground_truths)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, list):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])


# import os
# import traceback
# from functools import partial
# from pebble import ProcessPool
# from concurrent.futures import TimeoutError
# from typing import Union, List, Dict

# from verl.utils.import_utils import deprecated

# def process_single_item(
#     data_source, 
#     solution_str, 
#     ground_truth, 
#     extra_info,
#     sandbox_fusion_url,
#     concurrent_semaphore,
#     memory_limit_mb
# ):
#     if data_source == "openai/gsm8k":
#         from . import gsm8k
#         return gsm8k.compute_score(solution_str, ground_truth)
#     elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
#         from . import math_verify
#         return math_verify.compute_score(solution_str, ground_truth)
#     elif data_source == "math_dapo" or data_source.startswith("aime"):
#         from . import math_dapo
#         return math_dapo.compute_score(solution_str, ground_truth)
#     elif data_source in ["numina_aops_forum", "numina_synthetic_math", "numina_amc_aime", "numina_synthetic_amc", "numina_cn_k12", "numina_olympiads"]:
#         from . import prime_math
#         return prime_math.compute_score(solution_str, ground_truth)
#     elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
#         if sandbox_fusion_url:
#             from . import sandbox_fusion
#             return sandbox_fusion.compute_score(
#                 sandbox_fusion_url, 
#                 concurrent_semaphore, 
#                 memory_limit_mb, 
#                 solution_str, 
#                 ground_truth, 
#                 continuous=True
#             )
#         else:
#             from . import prime_code
#             return prime_code.compute_score(solution_str, ground_truth, continuous=True)
#     elif data_source in ["hiyouga/geometry3k"]:
#         from . import geo3k
#         return geo3k.compute_score(solution_str, ground_truth)
#     elif data_source in ["searchR1_nq", "searchR1_triviaqa", "searchR1_popqa", "searchR1_hotpotqa", "searchR1_2wikimultihopqa", "searchR1_musique", "searchR1_bamboogle"]:
#         from . import search_r1_like_qa_em
#         return search_r1_like_qa_em.compute_score(solution_str, ground_truth)
#     elif data_source in ["bailing_verify"]:
#         from . import bailing_verify
#         return bailing_verify.compute_score(solution_str, ground_truth)
#     else:
#         raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

# def default_compute_score(
#     data_sources: Union[str, List[str]],
#     solution_strs: Union[str, List[str]],
#     ground_truths: Union[str, List[str]],
#     extra_infos: Union[Dict, List[Dict]] = None,
#     sandbox_fusion_url: str = None,
#     concurrent_semaphore: int = None,
#     memory_limit_mb: int = None
# ) -> Union[float, Dict, List[Union[float, Dict]]]:
#     """Compute scores for given solutions based on the data sources using parallel processing."""
#     # Convert single input to list
#     if isinstance(data_sources, str):
#         data_sources = [data_sources]
#         solution_strs = [solution_strs]
#         ground_truths = [ground_truths]
#         extra_infos = [extra_infos] if extra_infos is not None else [None]
#         single_input = True
#     else:
#         single_input = False

#     if extra_infos is None:
#         extra_infos = [None] * len(data_sources)

#     # Calculate maximum number of worker processes
#     max_workers = min(max(os.cpu_count() - 32, 1), 128)  # Adjusted CPU usage

#     # Prepare arguments for each task
#     task_args = [
#         (data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb)
#         for data_source, solution_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos)
#     ]

#     results = []
#     with ProcessPool(max_workers=max_workers) as pool:
#         # Use partial to bind the function with fixed arguments
#         partial_func = partial(process_single_item)
        
#         # Submit tasks and handle results
#         future = pool.map(partial_func, *zip(*task_args), timeout=30)  # Added timeout parameter
#         iterator = future.result()
        
#         for i, task_arg in enumerate(task_args):
#             try:
#                 result = next(iterator)
#                 # Process result format
#                 if isinstance(result, dict):
#                     results.append(result)
#                 elif isinstance(result, (int, float, bool)):
#                     results.append(float(result))
#                 else:
#                     results.append(float(result[0]))
#             except TimeoutError as error:
#                 print(f"Task {i} timed out after {error.args[1]} seconds: {task_arg[0]}")
#                 results.append({"error": "timeout", "details": str(error)})
#             except NotImplementedError as error:
#                 print(f"Task {i} failed: {error}")
#                 results.append({"error": "not_implemented", "details": str(error)})
#             except Exception as error:
#                 print(f"Task {i} raised an unexpected error: {error}")
#                 traceback.print_exc()  # Print full traceback
#                 results.append({"error": "unexpected", "details": str(error)})

#     # Handle return value for single input
#     if single_input:
#         return results[0] if results else None
#     else:
#         return results

@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None, memory_limit_mb=None):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb)


__all__ = ["default_compute_score"]