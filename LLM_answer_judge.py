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
