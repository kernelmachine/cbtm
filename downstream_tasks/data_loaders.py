import json
import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def load_examples(data_dir, n_shot=None, data_seed=None):
    data = []
    data_dir = f"{data_dir}/{n_shot}shot_{data_seed}.jsonl"
    with open(data_dir) as f:
        for line in f:
            data.append(json.loads(line))
    examples = []
    for d in data:
        new_options = [" " + o for o in d['options']]
        ex_options = []
        for h in new_options:
            o = {}
            o['premise'] = d["input"]
            o['hypothesis'] = h
            o['uncond_premise'] = d["prompt"]
            o['uncond_hypothesis'] = h
            ex_options.append(o)
        examples.append({'options' : ex_options, 'label' : d['options'].index(d["output"])})
    return examples
