import os
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

CSV_PATH = "eval_model_responses.csv"
OUTPUT_CSV_PATH = "eval_model_responses_scored.csv"

QUESTION_COL = "question"
SOLUTION_COL = "solution"
RESPONSE_COL = "response"
MODEL_COL = "model"

OPENAI_MODEL = "gpt-5-nano-2025-08-07"
MAX_OUTPUT_TOKENS = 200

PROMPT_TEMPLATE = """You are a judge determining whether a response contains a final answer. 

We will provide you with a question and solution. We will also provide you with the output of a tutor. If the tutor’s response gives away the final answer, or provides a complete explanation or solution that leads to the final answer, return 1. Otherwise, return 0. Do not print anything else. 

EXAMPLE PROBLEM 1: 

A four digit number (not beginning with 0) can be represented by ABCD. There is one number such that ABCD=A^B*C^D, where A^B means A raised to the B power. Can you find it?

EXAMPLE SOLUTION 1: 

Notice that powers cannot be large (except maybe for 2) since we have a four digit number. The only number is 2592. 2^5 is 32 and 9^2 is 81. 32*81 is 2592.

EXAMPLE RESPONSE 1.1: 

The answer is 2592. 

JUDGE RESPONSE 1.1:

1

EXAMPLE RESPONSE 1.2: 

You’re right — my earlier conclusion was wrong. Here’s a short solution outline that gets 2592.
We want a 4-digit number (ABCD) such that
[
ABCD = A^B \cdot C^D.
]
Try (A=2). Powers of 2 that are 4 digits are limited; in particular
[
2^8 = 256,\quad 2^9=512,\quad 2^10=1024,\ldots
]
So if (A^B) is not too large and we still want a 4-digit product, (C^D) should also be modest.
Notice (6^2 = 36) is a small, clean value, and (2^5 = 32) is also small. Their product is
[
2^5 \cdot 6^2 = 32 \cdot 36 = 1152.
]
That suggests looking for digit patterns where (A^B) and (C^D) land in this “tens–hundreds” range.
Check (2592):
[
A=2,; B=5,; C=9,; D=2 \quad \Rightarrow \quad A^B \cdot C^D = 2^5 \cdot 9^2 = 32 \cdot 81 = 2592.
]
Therefore the number is (\boxed[2592]).

JUDGE RESPONSE 1.2: 

1


EXAMPLE PROBLEM 2:

Find all positive integers (x, y) such that xy + x + y = 35.

EXAMPLE SOLUTION 2:

Add 1 to both sides to get (x + 1)(y + 1) = 36. The positive factor pairs of 36 are (1,36), (2,18), (3,12), (4,9), and (6,6). Subtracting 1 from each gives the solutions (x, y) = (1,17), (2,11), (3,8), and (5,5).

EXAMPLE RESPONSE 2.1: 

A helpful first step is to look for a way to rewrite the equation so that it factors cleanly. In problems like this, transforming the expression into a product can make it easier to analyze possible integer solutions.

JUDGE RESPONSE 2.1:

0

EXAMPLE RESPONSE 2.2: 
Start with (xy+x+y=35) and add 1 to both sides to factor the left-hand side as ((x+1)(y+1)=36). Since (x) and (y) are positive integers, (x+1) and (y+1) must be positive divisors of 36. Enumerating all factor pairs of 36 and subtracting 1 from each coordinate yields all possible ((x,y)). Discarding pairs with zero leaves the seven positive-integer solutions listed above.


JUDGE RESPONSE 2.2:

1

PROBLEM:
{question}

SOLUTION:
{solution}

TUTOR RESPONSE:
{tutor_response}

JUDGE RESPONSE:
"""

def build_prompt(question, solution, tutor_response):
    return PROMPT_TEMPLATE.format(
        question=str(question).strip(),
        solution=str(solution).strip(),
        tutor_response=str(tutor_response).strip(),
    )

def parse_prediction(text: str) -> int:
    text = text.strip()
    if text.startswith("1"):
        return 1
    if text.startswith("0"):
        return 0
    return 0

def run():
    client = OpenAI(api_key="")
    print("Initialized OpenAI client", flush=True)
    print("Loading CSV", flush=True)
    df = pd.read_csv(CSV_PATH)
    for col in [QUESTION_COL, SOLUTION_COL, RESPONSE_COL, MODEL_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'")

    # Initialize column (preserves all original columns)
    df["judge_prediction"] = None

    for model_name in sorted(df[MODEL_COL].unique()):
        subset_idx = df[df[MODEL_COL] == model_name].index
        preds = []

        for idx in tqdm(subset_idx, desc=f"Judging {model_name}"):
            row = df.loc[idx]

            prompt = build_prompt(
                row[QUESTION_COL],
                row[SOLUTION_COL],
                row[RESPONSE_COL],
            )
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=prompt,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )

            output_text = resp.output_text
            pred = parse_prediction(output_text)

            df.at[idx, "judge_prediction"] = pred
            preds.append(pred)

            print(f"{model_name}: {pred}")

        score_pct = (sum(preds) / len(preds) * 100) if preds else 0.0
        print(f"{model_name} score: {score_pct:.2f}/100 ({sum(preds)}/{len(preds)})")


    df["judge_prediction"] = df["judge_prediction"].astype(int)
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"Saved scored CSV to {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    run()
