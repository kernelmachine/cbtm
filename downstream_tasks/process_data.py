import argparse
import json
from pathlib import Path
import MetaICL

TASK2PROMPT = {"glue-sst2": " My sentiment is", \
               "financial_phrasebank": " It is",\
               "amazon_polarity": " It is",\
               "ag_news": " The topic is",\
                "dbpedia_14": " The topic of the paragraph is",\
                "tweet_eval-offensive": " Is the sentence hate or non-offensive?",\
                }


def read_data(task, n_sample, random_seed, data_path, split):
    # read data
    demos = []
    with open(f"{data_path}/{task}_{n_sample}_{random_seed}_train.jsonl") as f:
        for line in f:
            ex = json.loads(line)
            ex["options"] = [e.lower() for e in ex["options"]]
            ex["output"] = ex["output"].lower()
            demos.append(ex)
    tests = []
    with open(f"{data_path}/{task}_{n_sample}_{random_seed}_{split}.jsonl") as f:
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

def write_data(demos, tests, k, task, seed, write_path, split):
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
    with open(f"{write_path}/{task}/{split}/{k}shot_{seed}.jsonl", "w") as f:
        for test_ex in new_tests:
            f.write(json.dumps(test_ex))
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_args('--random-seed', type=list, nargs='+')
    parser.add_args('--n-shot', typ=list, nargs='+')
    parser.add_args('--data-write-path', type=str)
    parser.add_args('--data-read-path', type=str)
    parser.add_args('--tasks', type=list, nargs='+')
    parser.add_args('--n-examples', type=int, default=1000)
    parser.add_args('--split', type=str, default='test')
    
    args = parser.add_args()

    tasks = TASK2PROMPT.keys() if args.tasks is None else args.tasks
    k_options = [int(n) for n in args.n_shot]
    random_seeds = [int(seed) for seed in args.random_seed]
    for random_seed in random_seeds:
        for task in tasks:
            for k in k_options:
                data_path = f"{args.data_read_path}/{task}"
                demos, tests = read_data(task, args.n_examples, random_seed, args.split)
                write_data(demos, tests, k, task, random_seed, args.data_write_path, args.split)
