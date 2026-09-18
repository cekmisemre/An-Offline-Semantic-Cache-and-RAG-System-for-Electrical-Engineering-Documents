"""MMLU (electrical_engineering) evaluation for the local model.

Why this is here
----------------
Everything else I measured is about the system: does it retrieve the right passage and
answer from it. This measures something different - how much electrical engineering the
local model already knows on its own, with no documents involved.

That is worth measuring because of the project's central finding: when a document
disagrees with what the model already believes about electrical practice, the model
follows its own knowledge, and no instruction I tried could stop it. This benchmark puts
a number on the knowledge doing the overriding.

MMLU is a standard, widely cited benchmark (Hendrycks et al., ICLR 2021). The
electrical_engineering subset is 4-choice multiple choice, so grading is exact - unlike
my own benchmark, which grades by string matching.

Protocol
--------
Zero-shot, which must be stated because it is not the only option: the original paper
reports 5-shot results, and 5-shot scores are usually higher. So these numbers are not
directly comparable to the leaderboard in the MMLU repository. The honest comparison
point here is the random baseline of 25%.

Two system prompts are measured, because the difference is itself a result:
  neutral    - a plain instruction, measuring what the model knows
  EE domain  - the exact system prompt my assistant uses, showing whether restricting
               the assistant to one domain costs anything on in-domain questions

Usage
-----
    python mmlu_benchmark.py --data <path to the unpacked data folder>
    python mmlu_benchmark.py --data ... --limit 20      (quick check first)
"""

import argparse
import csv
import math
import os
import re
import sys
import time

from openai import OpenAI

CHOICES = ["A", "B", "C", "D"]
SUBJECT = "electrical_engineering"

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

NEUTRAL_PROMPT = (
    "You are answering a multiple-choice exam question. Reply with a single letter: "
    "A, B, C or D. Do not explain."
)

# Copied verbatim from the frozen pipeline so the comparison is honest.
EE_DOMAIN_PROMPT = (
    "You are a helpful assistant specialized ONLY in electrical engineering technical "
    "knowledge retrieval - helping engineers and technicians understand electrical "
    "standards, technical manuals, and regulations such as wiring design, overcurrent "
    "protection, grounding, and equipment installation requirements. Answer strictly "
    "using the information provided to you; if the provided information doesn't cover "
    "the question, say so rather than guessing. If a question falls outside this domain, "
    "say briefly that it's outside your area of expertise and do not attempt to answer "
    "it. Answer accurately and exclusively in ENGLISH. Keep answers professional and "
    "concise."
)


def find_test_file(data_dir):
    """Locates the subject's test CSV, tolerating the two layouts the archive unpacks
    into depending on how it was extracted."""
    candidates = [
        os.path.join(data_dir, "test", f"{SUBJECT}_test.csv"),
        os.path.join(data_dir, "data", "test", f"{SUBJECT}_test.csv"),
        os.path.join(data_dir, f"{SUBJECT}_test.csv"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise SystemExit(
        f"Could not find {SUBJECT}_test.csv under '{data_dir}'.\n"
        f"Looked in:\n  " + "\n  ".join(candidates) +
        "\nPass the folder that contains the 'test' directory with --data."
    )


def load_questions(path, limit=None):
    """MMLU CSVs have no header: question, A, B, C, D, correct letter."""
    questions = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            questions.append({
                "question": row[0].strip(),
                "options": [c.strip() for c in row[1:5]],
                "answer": row[5].strip().upper(),
            })
    if limit:
        questions = questions[:limit]
    return questions


def build_prompt(item):
    lines = [item["question"], ""]
    for letter, option in zip(CHOICES, item["options"]):
        lines.append(f"{letter}. {option}")
    lines.append("")
    lines.append("Answer with one letter only.")
    return "\n".join(lines)


def extract_letter(reply):
    """Takes the first standalone A-D in the reply. Returns None if there is none, so
    unparseable replies can be counted separately instead of silently scored wrong."""
    if not reply:
        return None
    match = re.search(r"\b([ABCD])\b", reply.strip().upper())
    return match.group(1) if match else None


def wilson(correct, total, z=1.96):
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def run(questions, system_prompt, label):
    correct = 0
    unparseable = 0
    refusals = 0
    start = time.time()

    print(f"\n[{label}] {len(questions)} questions")
    for i, item in enumerate(questions, start=1):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": build_prompt(item)},
                ],
                temperature=0.0,     # deterministic: this measures knowledge, not sampling
                max_tokens=8,        # a letter is all that is needed
            )
            reply = response.choices[0].message.content or ""
        except Exception as error:
            print(f"  question {i}: request failed ({error})")
            unparseable += 1
            continue

        if re.search(r"outside|expertise|cannot|can't|sorry", reply, re.IGNORECASE):
            refusals += 1

        letter = extract_letter(reply)
        if letter is None:
            unparseable += 1
        elif letter == item["answer"]:
            correct += 1

        if i % 25 == 0:
            print(f"  {i}/{len(questions)} done, {correct} correct so far")

    elapsed = time.time() - start
    low, high = wilson(correct, len(questions))
    print(f"  accuracy    : {correct}/{len(questions)} = {correct / len(questions) * 100:.1f}%"
          f"   95% CI [{low * 100:.1f}%, {high * 100:.1f}%]")
    print(f"  unparseable : {unparseable}")
    print(f"  refusals    : {refusals}")
    print(f"  time        : {elapsed:.0f}s ({elapsed / len(questions):.2f}s per question)")
    return {"correct": correct, "total": len(questions), "unparseable": unparseable,
            "refusals": refusals, "seconds": elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data",
                        help="folder holding the unpacked MMLU archive")
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N questions (for a quick check)")
    args = parser.parse_args()

    path = find_test_file(args.data)
    questions = load_questions(path, args.limit)
    print(f"Loaded {len(questions)} questions from {path}")
    print("Protocol: zero-shot, temperature 0. Random baseline is 25%.")

    results = {
        "neutral prompt": run(questions, NEUTRAL_PROMPT, "neutral prompt"),
        "EE domain prompt": run(questions, EE_DOMAIN_PROMPT, "EE domain prompt (the assistant's own)"),
    }

    print("\n" + "=" * 64)
    print(f"{'configuration':24}{'accuracy':>12}{'95% CI':>20}{'refusals':>10}")
    print("-" * 64)
    for label, r in results.items():
        low, high = wilson(r["correct"], r["total"])
        accuracy = f"{r['correct']}/{r['total']} = {r['correct'] / r['total'] * 100:.1f}%"
        print(f"{label:24}{accuracy:>12}{f'[{low * 100:.1f}%, {high * 100:.1f}%]':>20}{r['refusals']:>10}")
    print(f"{'random baseline':24}{'25.0%':>12}")
    print("=" * 64)


if __name__ == "__main__":
    main()
