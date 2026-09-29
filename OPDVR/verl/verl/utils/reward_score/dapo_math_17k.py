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
from verl.workers.config import RolloutConfig
import re
from verl.utils.config import omega_conf_to_dataclass
_SOLUTION_CLIP_CHARS = 100


def extract_solution(solution_str, method="flexible"):
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        # only match digit strings that end in a digit (commas and a decimal point allowed), e.g. 123, 1,234.56, -78.9, but not 123.
        answer = re.findall(r"-?[0-9][0-9,]*\.?[0-9]*", solution_str)
        # drop empty matches and matches not ending in a digit, and strip the thousands separators
        answer = [a.replace(',', '') for a in answer if a and a[-1].isdigit()]
        return answer


def compute_score(solution_str, ground_truth, extra_info, method="flexible", format_score=0.0, score=1.0, length_score_coef=0.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    
    # make sure response_length is a float (the reward manager does not always pass it; default 0, unused below)
    response_length = extra_info.get("response_length", 0)
    if hasattr(response_length, 'item'):
        response_length = float(response_length.item())
    else:
        response_length = float(response_length)
    
    max_response_length = 2048
   
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None or (isinstance(answer, list) and len(answer) == 0):
        return format_score 
    def safe_float(x):
        try:
            return float(x)
        except Exception:
            return None
    gt_val = safe_float(ground_truth)
    if gt_val is None:
        # ground_truth is not numeric, fall back to exact string comparison
        if isinstance(answer, list):
            if ground_truth in answer:
                return score 
            else:
                return format_score 
        else:
            if answer == ground_truth:
                return score 
            else:
                return format_score 
    # ground_truth is numeric, compare every answer as a float
    if isinstance(answer, list):
        for a in answer:
            a_val = safe_float(a)
            if a_val is not None and a_val == gt_val:
                
                return score 
        return format_score 
    else:
        a_val = safe_float(answer)
        if a_val is not None and a_val == gt_val:
           
            return score 
        else:
            return format_score


def compute_score_with_entropy(
    solution_str,
    ground_truth,
    extra_info=None,
    method: str = "flexible",
    format_score: float = 0.0,
    score: float = 1.0,
    entropy_coef: float = 0.01,
    last_token_entropy_coef: float = 0.0,
):
    """GSM8K scoring + entropy regularization.

    Args:
        solution_str: the generated text
        ground_truth: the reference answer
        extra_info: per-sample extra info (a dict, or one entry of a list); expected to carry 'entropy_mean' / 'entropy_last'
        method: answer-parsing mode ('flexible' or 'strict')
        format_score: score when the format does not match
        score: base score when the answer matches
        entropy_coef: bonus coefficient on the mean entropy
        last_token_entropy_coef: bonus coefficient on the last-token entropy
    Returns:
        float: reward including the entropy bonus
    """
    base = compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        method=method,
        format_score=format_score,
        score=score,
    )

    if extra_info is None:
        return float(base)

    # accept extra_info as a single dict or a List[dict] (the reward manager usually passes one dict per sample)
    if isinstance(extra_info, list):
        # if a list arrives, take its first entry
        if len(extra_info) > 0 and isinstance(extra_info[0], dict):
            extra_info = extra_info[0]
        else:
            return float(base)

    entropy_mean = extra_info.get("entropy_mean", 0.0)
    entropy_last = extra_info.get("entropy_last", 0.0)

    try:
        entropy_mean_val = float(entropy_mean)
    except Exception:
        entropy_mean_val = 0.0
    try:
        entropy_last_val = float(entropy_last)
    except Exception:
        entropy_last_val = 0.0

    bonus = entropy_coef * entropy_mean_val + last_token_entropy_coef * entropy_last_val
    # final_score = float(base + bonus)
    final_score = base + bonus
    # --- Logging block: append solution_str, ground_truth, extra_info, and final_score to a log file ---
    # GSM8K_ENTROPY_LOG_PATH sets the log path; the default is gsm8k_entropy_log.jsonl in the working directory
    try:
        import os, json, time
        log_path = os.environ.get("GSM8K_ENTROPY_LOG_PATH", "gsm8k_entropy_log.jsonl")
        log_item = {
            "ts": time.time(),
            "solution_str": solution_str,
            "ground_truth": ground_truth,
            "entropy_mean": entropy_mean_val,
            "entropy_last": entropy_last_val,
            "base_score": float(base),
            "final_score": final_score,
            "method": method,
        }
        # keep any other fields of extra_info as well
        if isinstance(extra_info, dict):
            for k, v in extra_info.items():
                if k not in log_item:
                    log_item[k] = v
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_item, ensure_ascii=False) + "\n")
    except Exception as e:
        # fail silently so training is not interrupted; switch to print when debugging
        pass

    return final_score


__all__ = [
    "extract_solution",
    "compute_score",
    "compute_score_with_entropy",
]

