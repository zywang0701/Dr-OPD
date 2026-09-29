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
# from . import gsm8k, math, prime_math, prime_code

import os

from verl.utils.import_utils import deprecated


def code_compute_score(
    solution_str,
    ground_truth,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """Scoring for the code datasets (taco/codecontests/apps/codeforces): 1.0 only when every test case passes.

    ground_truth is a JSON string {"inputs": [...], "outputs": [...]}; prime_code extracts the fenced python block
    from the generation and runs it on stdin in a subprocess; the per-case timeout is CODE_TEST_TIMEOUT (default 5 s).
    CODE_CONTINUOUS=1 gives partial credit (first 10 cases only, 10 s each); off by default.
    """
    continuous = os.environ.get("CODE_CONTINUOUS", "0") == "1"
    if sandbox_fusion_url:
        from . import sandbox_fusion

        return sandbox_fusion.compute_score(
            sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=continuous
        )
    from . import prime_code

    return prime_code.compute_score(solution_str, ground_truth, continuous=continuous)


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    # Explicit whitelist instead of the original catch-all check on a single data_source,
    #   which sent every other data_source into prime_math_refine and left all the branches below (including the code branch) unreachable.
    #   Consequence: code responses carry no boxed answer, so the math scorer silently returned a constant 0.
    #   Now the math datasets we use still go to prime_math_refine (identical numbers) and the code datasets go to prime_code / the sandbox,
    #   while an unlisted data_source keeps walking the original branch chain and raises NotImplementedError if nothing matches.
    MATH_REFINE_SOURCES = ("DeepMath-103K", "math", "olympiad_bench", "minerva", "amc", "aime", "aime25")
    CODE_SOURCES = ("codecontests", "apps", "codeforces", "taco")
    if data_source in MATH_REFINE_SOURCES:
        from . import prime_math_refine
        res = prime_math_refine.compute_score(solution_str, ground_truth)
    elif data_source in CODE_SOURCES:
        res = code_compute_score(
            solution_str, ground_truth, sandbox_fusion_url=sandbox_fusion_url,
            concurrent_semaphore=concurrent_semaphore, memory_limit_mb=memory_limit_mb,
        )
    elif data_source == "openai/gsm8k":
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source == "open-r1/DAPO-Math-17k-Processed":
        from . import dapo_math_17k
        res = dapo_math_17k.compute_score(solution_str, ground_truth, extra_info)
    elif data_source in ["HuggingFaceH4/aime_2024", "rawsh/aime_2025","math-ai/amc23"]:
        from . import aime
        res = aime.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from . import math_reward

        res = math_reward.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in ["math_dapo", "math", "math_dapo_reasoning"] or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


__all__ = ["default_compute_score"]
