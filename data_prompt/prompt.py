import re
import json
import openai
import argparse
from tqdm import tqdm
from time import sleep
from copy import deepcopy
import random

def preprocess(path):
    data_true = []
    data_false = []
    for line in open(path):
        info = json.loads(line.strip())
        if info['label']:
            data_true.append(info)
        else:
            data_false.append(info)
    return data_true, data_false

def construct_prompt(ex, k_shot_exs):
    instruction = "Follow the examples and generate conclusion for a given premises: "
    for i in range(len(k_shot_exs)):
        instruction = instruction + " " + str(i) +" $premises$: " + k_shot_exs[i]["premises"] + " $conclusion$: " + k_shot_exs[i]["conclusion"]
    instruction += " $premises$: " + ex["premises"] + " $conclusion$: "
    return instruction

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--k", type=int, default=5, help="k-shot for ICL learning")
    parser.add_argument(
        "--sleep",
        type=int,
        default=2,
        help="Time gap (in seconds) between two consecutive requests. Only necessary for Codex. (Default: 15)",
    )
    parser.add_argument("--api-key", type=str, default="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", help="OpenAI API key.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="./entailment_verifier_task2_samples.txt",
        help="Path to the training data.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default='./tmp.jsonl',
        help="Path to the output file.",
    )
    args = parser.parse_args()

    data_true, data_false = preprocess(args.data_path)
    openai.api_key = args.api_key

    f = open(args.out_path, "a+")
    for ex in tqdm(data_false[18:]):
        k_shot_exs = random.sample(data_true, args.k)
        prompt = construct_prompt(ex, k_shot_exs)
        ret = deepcopy(ex)
        try:
            response = openai.ChatCompletion.create(
                model=args.model,
                messages=[{"role": "user",
                           "content": prompt}],
                temperature=0,
                max_tokens=256
            )
            sleep(args.sleep)

            ex['conclusion'] = response["choices"][0]["message"]["content"]
            ex['label'] = True
            json_ret = json.dumps(ex)
            f.write(json_ret + '\n')
            print(ex["premises"] + "\t" + ex['conclusion'])
        except:
            print("request error")
            sleep(60)
    f.close()


if __name__ == "__main__":
    main()
