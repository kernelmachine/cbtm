import os
import json
from pathlib import Path

TASK2PROMPT = {"glue-sst2": " My sentiment is"}


def read_data(task, n_sample, random_seed, data_path):
    # read data
    demos = []
    with open(f"{data_path}/{task}_8_{random_seed}_train.jsonl") as f:
        for line in f:
            ex = json.loads(line)
            ex["options"] = [e.lower() for e in ex["options"]]
            ex["output"] = ex["output"].lower()
            demos.append(ex)
    tests = []
    with open(f"{data_path}/{task}_8_{random_seed}_dev.jsonl") as f:
        for line in f:
            ex = json.loads(line)
            ex["options"] = [e.lower() for e in ex["options"]]
            ex["output"] = ex["output"].lower()
            tests.append(ex)
    tests = tests[:n_sample]
    return demos, tests

def concat_prompt(ex):
    return ex["input"] + TASK2PROMPT[task]

def concat_prompt_ans(ex):
    return ex["input"] + TASK2PROMPT[task] + " " + ex["output"]

def write_data(demos, tests, k, task, seed, write_path):
    Path(write_path).mkdir(parents=True, exist_ok=True)
    new_tests = []
    for test_ex in tests:
        test_ex_after_prompt =  concat_prompt(test_ex)
        if k == 0:
            test_ex["input"] = test_ex_after_prompt
        else:
            demos_after_prompt = [concat_prompt_ans(demo_ex) for demo_ex in demos][:k]
            test_ex["input"] = "\n\n".join(demos_after_prompt) + f"\n\n{test_ex_after_prompt}"
        test_ex["prompt"] = TASK2PROMPT[task]
        new_tests.append(test_ex)
    Path(f"{write_path}/{task}").mkdir(parents=True, exist_ok=True)
    with open(f"{write_path}/{task}/{k}shot_{seed}.jsonl", "w") as f:
        for test_ex in new_tests:
            f.write(json.dumps(test_ex))
            f.write("\n")


if __name__ == "__main__":
    n_sample = 1000
    all_task = ["glue-sst2"] 
    k_options = [8]
    random_seeds = [0] 
    write_path = "./"
    data_path = "/path/to/data"
    for random_seed in random_seeds:
        for task in all_task:
            for k in k_options:
                data_path = f"{data_path}/{task}"
                demos, tests = read_data(task, n_sample, random_seed)
                write_data(demos, tests, k, task, random_seed, write_path)

