import argparse
import copy
import json
from pathlib import Path
from configs import PS_NAME_DICT
# from templates import TASK2PROMPT

TASK2PROMPT = {"glue-sst2": " My sentiment is", \
               "financial_phrasebank": " It is",\
               "amazon_polarity": " It is",\
               "ag_news": " The topic is",\
                "dbpedia_14": " The topic of the paragraph is",\
                "tweet_eval-offensive": " Is the sentence hate or non-offensive?",\
                }

def load_wanted_prompt(task, prompt_name=None, no_prompt=False, use_hardcoded_prompt=False):
    if no_prompt or use_hardcoded_prompt:
        return None, None
    
    from promptsource.templates import TemplateCollection
    ps_task, ps_subtask, ps_prompt_name = PS_NAME_DICT.get(task, (task, None, ""))
    chosen_prompt_name = prompt_name if prompt_name else ps_prompt_name
    collection = TemplateCollection()
    prompts = collection.get_dataset(ps_task, ps_subtask)

    if not chosen_prompt_name: 
        all_prompt_names = [name for name in prompts.all_template_names if "no_option" not in name]
        keywords = ["multiple_choice", "most_correct", "most_suitable"]
        _all_prompt_names = [name for name in all_prompt_names if any([keyword in name for keyword in keywords])]
        if len(_all_prompt_names) > 0:
            all_prompt_names = _all_prompt_names
        elif len(all_prompt_names) == 0:
            raise Error(f'No template found for {task}')
        chosen_prompt_name = all_prompt_names[0] # TODO @margaretli is there another way to select 

    return chosen_prompt_name, prompts[chosen_prompt_name]


def apply_promptsource(prompt, example, is_test=True):
        # result = prompt.apply(example)
        input, output = prompt.apply(example)
        if is_test:
            example_copy = copy.deepcopy(example)
            example_copy['output'] = ""
            input, _ = prompt.apply(example_copy)
        options = prompt.get_answer_choices_list(example)

        # these are for special cases where prompt does not handle answer options properly
        if task=="commonsense_qa":
            assert options is None
            options = example["choices"]["text"]
        elif task=="codah":
            assert options is None
            output = output.strip()
            options = [o.strip() for o in example["candidate_answers"]]
        elif task=="yelp_polarity":
            assert options == ["no", "yes"] and output in ["yes.", "no."], (output, options)
            output = output[:-1]
        elif task=="sick":
            assert options is None
            options = ["entailment", "neutral", "contradiction"]
        assert output in options, (task, output, options)
        return {"task": task, "input_": input, "output": output, "options": options}


def read_data(task, n_sample, k, random_seed, data_path, split):
    # read data
    demos = []
    with open(f"{data_path}/{task}_{k}_{random_seed}_train.jsonl") as f:
        for line in f:
            ex = json.loads(line)
            ex["options"] = [e.lower() for e in ex["options"]]
            ex["output"] = ex["output"].lower()
            demos.append(ex)
    tests = []
    with open(f"{data_path}/{task}_{k}_{random_seed}_{split}.jsonl") as f:
        for line in f:
            ex = json.loads(line)
            ex["options"] = [e.lower() for e in ex["options"]]
            ex["output"] = ex["output"].lower()
            tests.append(ex)
    tests = tests[:n_sample]
    return demos, tests

# def concat_prompt(ex, use_hardcoded_prompt):
#     if use_hardcoded_prompt:
#         return ex["input"] + "\n" + TASK2PROMPT[task] + " "
#     return ex["input"] + "\n"

# def concat_prompt_ans(ex, use_hardcoded_prompt):
#     if use_hardcoded_prompt:
#         return ex["input"] + "\n" + TASK2PROMPT[task] + " " + ex["output"]
#     return ex["input"] + "\n" + ex["output"]
   

def apply_prompts_and_write_data(
        demos, tests, k, original_task, seed, write_path, split, use_hardcoded_prompt=False, prompt=None, prompt_name=None
    ):
    demos = demos[:k]

    if prompt:
        [ex.update(apply_promptsource(prompt, ex)) for ex in demos]
        [ex.update(apply_promptsource(prompt, ex, is_test=True)) for ex in tests]
        task = f"inst:{original_task}:{prompt_name}"
        prompt_text = apply_promptsource({})
    elif use_hardcoded_prompt:
        hardcoded_prompt = TASK2PROMPT[original_task]
        [ex.update({'input': hardcoded_prompt.format(input=ex['input'], output=ex['output'])}) for ex in demos]
        [ex.update({'input': hardcoded_prompt.format(input=ex['input'], output="")}) for ex in tests]
        task = f"hard_inst:{original_task}"
        prompt_text = hardcoded_prompt.format(input="", output="")
    else:
        task = original_task
        prompt_text = ""
    
    demos_input = "\n\n".join([ex["input"] for ex in demos])
    for test_ex in tests:
        test_ex["input"] = demos_input + "\n\n" + test_ex["input"]
        test_ex["prompt"] = prompt_text

    Path(f"{write_path}/{task}/{split}").mkdir(parents=True, exist_ok=True)
    with open(f"{write_path}/{task}/{split}/{k}shot_{seed}.jsonl", "w") as f:
        for test_ex in tests:
            f.write(json.dumps(test_ex))
            f.write("\n")    


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', type=int, nargs='+')
    parser.add_argument('--n-shot', type=int, nargs='+')
    parser.add_argument('--data-write-path', type=str)
    parser.add_argument('--data-read-path', type=str)
    parser.add_argument('--tasks', type=str, nargs='+')
    parser.add_argument('--n-examples', type=int, default=1000)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--use-hardcoded-prompt', action='store_true')
    parser.add_argument('--no-prompt', action='store_true')
    parser.add_argument('--prompt-name', type=str, default=None)
    args = parser.parse_args()

    if args.no_prompt and (args.use_hardcoded_prompt or args.template_name):
        raise Error("If --no-prompt, cannot specify --prompt-name or --use-hardcoded-prompt")
    elif args.no_prompt and args.use_hardcoded_prompt:
        raise Error("Can only specify one of --prompt-name and --use-hardcoded-prompt")

    tasks = PS_NAME_DICT.keys() if args.tasks is None else args.tasks
    k_options = [int(n) for n in args.n_shot]
    random_seeds = [int(seed) for seed in args.random_seed]
    for task in tasks:
        prompt_name, prompt = load_wanted_prompt(task, args.prompt_name, args.no_prompt, args.use_hardcoded_prompt)
        for random_seed in random_seeds:
            for k in k_options:
                data_path = f"{args.data_read_path}/{task}"
                demos, tests = read_data(task, args.n_examples, k, random_seed, data_path, args.split)
                # write_data(demos, tests, k, task, random_seed, args.data_write_path, args.split, args.use_hardcoded_prompt)
                # apply_prompts_and_write_data(
                #     demos, tests, k, task, random_seed, 
                #     args.data_write_path, args.split, 
                #     use_hardcoded_prompt=args.use_hardcoded_prompt, 
                #     prompt=prompt, prompt_name=prompt_name
                # )