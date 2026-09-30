# -*- coding: utf-8 -*-
"""4-shot base-model prompt + its stopping rule, ported verbatim from opd_trust
(opd_trust/util/prompts.py and opd_trust/util/stopping.py, the E015-E025 campaign).
Used only when the few-shot prompt mode is enabled: data.raw_prompt_text=True feeds the
prompt text below as-is (no chat template) and rollout.fewshot_stop=True applies
cut_completion to every generated response (train and validation).

Stopping rule (earliest wins): (i) EOS, kept; (ii) first \\boxed{...} closed, the
token holding the closing brace kept; (iii) the next-problem delimiter "\\nProblem:",
its first token dropped; (iv) the length cap. The cut is post-hoc on the token ids,
so vLLM's own stop string only saves compute; the prefix distribution is unchanged."""
from typing import List, Tuple

# ---------------------------------------------------------------- prompt (prompts.py)
INSTRUCT_HEADER = (
    "Solve the following math problems. Reason step by step, "
    "and put your final answer within \\boxed{}.\n\n")

FEWSHOT = [
    ("What is $1+2+3+\\cdots+10$?",
     "The sum of the first $n$ positive integers is $\\frac{n(n+1)}{2}$. "
     "For $n=10$ this is $\\frac{10\\cdot 11}{2}=55$. "
     "The final answer is $\\boxed{55}$."),
    ("Solve for $x$: $2x+3=11$.",
     "Subtracting $3$ from both sides gives $2x=8$. "
     "Dividing both sides by $2$ gives $x=4$. "
     "The final answer is $\\boxed{4}$."),
    ("A rectangle has length $8$ and width $5$. What is its area?",
     "The area of a rectangle is length times width, so it is "
     "$8\\times 5=40$. The final answer is $\\boxed{40}$."),
    ("A fair coin is flipped twice. What is the probability that both flips are heads?",
     "The flips are independent, each heads with probability $\\frac{1}{2}$. "
     "So both are heads with probability $\\frac{1}{2}\\cdot\\frac{1}{2}=\\frac{1}{4}$. "
     "The final answer is $\\boxed{\\frac{1}{4}}$."),
]

FEWSHOT_STOP = "\nProblem:"


def build_student_prompt(question: str) -> str:
    parts = [INSTRUCT_HEADER]
    for q, s in FEWSHOT:
        parts.append(f"Problem:\n{q}\n\nSolution:\n{s}\n\n")
    parts.append(f"Problem:\n{question}\n\nSolution:\n")
    return "".join(parts)


# ---------------------------------------------------------------- stopping (stopping.py)
def _bytes_to_unicode():
    """GPT-2 byte<->unicode table."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD))
          + list(range(0xAE, 0x100)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_U2B = {u: b for b, u in _bytes_to_unicode().items()}


class TokenByteMap:
    """token id -> exact bytes (lazy cache; special tokens map to their literal text)."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.special_ids = set(tokenizer.all_special_ids)
        self._cache = {}

    def token_bytes(self, tid: int) -> bytes:
        b = self._cache.get(tid)
        if b is None:
            s = self.tok.convert_ids_to_tokens(int(tid))
            if s is None:
                b = b""          # vocab hole (model head > tokenizer entries): no bytes
            elif tid in self.special_ids:
                b = s.encode("utf-8")
            else:
                try:
                    b = bytes(_U2B[ch] for ch in s)
                except KeyError:
                    b = self.tok.decode([int(tid)]).encode("utf-8")
            self._cache[tid] = b
        return b


def _find_box_close(data: bytes) -> int:
    """Byte position right after the '}' closing the first \\boxed{...}; -1 if not closed."""
    i = data.find(b"\\boxed")
    if i == -1:
        return -1
    j = data.find(b"{", i)
    if j == -1:
        return -1
    depth = 0
    for k in range(j, len(data)):
        if data[k:k + 1] == b"{":
            depth += 1
        elif data[k:k + 1] == b"}":
            depth -= 1
            if depth == 0:
                return k + 1
    return -1


def cut_completion(byte_map: TokenByteMap, completion_ids: List[int],
                   eos_ids, delimiters: List[str], cap: int,
                   box_cut: bool = True) -> Tuple[List[int], dict]:
    """Returns (kept token prefix, info); info.stop_reason in {eos, box, delimiter, cap, end}."""
    pieces = [byte_map.token_bytes(t) for t in completion_ids]
    ends = []
    total = 0
    for p in pieces:
        total += len(p)
        ends.append(total)
    data = b"".join(pieces)

    events = []                                 # (byte position, tokens kept, reason)
    for k, t in enumerate(completion_ids):      # (i) EOS: kept
        if int(t) in eos_ids:
            events.append((ends[k], k + 1, "eos"))
            break
    bpos = _find_box_close(data) if box_cut else -1   # (ii) box closed: token kept
    if bpos != -1:
        k = next(i for i, e in enumerate(ends) if e >= bpos)
        events.append((bpos, k + 1, "box"))
    for d in delimiters:                        # (iii) delimiter: its first token dropped
        db = d.encode("utf-8")
        dpos = data.find(db)
        if dpos == 0 or (dpos == -1 and d.startswith("\n")
                         and data.startswith(db.lstrip(b"\n"))):
            dpos = 0                            # starts a new problem right away
        if dpos != -1:
            k = next((i for i, e in enumerate(ends) if e > dpos), len(ends))
            events.append((dpos, k, "delimiter"))

    if events:
        _, keep, reason = min(events, key=lambda x: (x[0], x[1]))
    else:
        keep = len(completion_ids)
        reason = "cap" if len(completion_ids) >= cap else "end"
    kept = list(completion_ids[:keep])
    info = {"stop_reason": reason,
            "truncated": reason == "cap",
            "box_closed": bpos != -1 and reason in ("box", "eos"),
            "n_raw": len(completion_ids), "n_kept": keep}
    return kept, info
