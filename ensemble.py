from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import random
import numpy as np
import torch
import os
import pdb
from collections import defaultdict
from scipy.special import softmax
from pathlib import Path
from collections import Counter
import json
import pandas as pd
from ipdb import set_trace as bp


'''
acc
'''
def compute_acc(predictions_list):
    labels = [pred['label'] for pred in predictions_list]
    # get predictions into list by scoring key
    predictions_dict = {key:list(map(lambda v: v[key], predictions_list)) for key in predictions_list[0].keys()}

    # calculate accuracies
    results = {key: sum(list(map(lambda v: v[0] == v[1], zip(predictions_dict[key] , labels))))/len(labels) for key in predictions_dict.keys()}

    # save labels for later
    predictions_dict['labels'] = labels
    return results, predictions_dict, predictions_list


def zero_out(output, k):
    output = torch.FloatTensor(output)
    vals, idx = output.topk(k)
    topk = torch.zeros_like(output)
    topk[idx] = vals
    topk = topk.numpy()

    # topk = np.zeros((topk.shape[0]))
    # topk[7] = 1
    return topk, idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('--model', type=str, default='xl')
    parser.add_argument('--softmax', type=int, default=0)
    parser.add_argument('--coeffs', type=int, default=0)
    parser.add_argument('--output', type=str, default='./')
    parser.add_argument('--n-shot', type=int, default=0)
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--variant', type=int, default=None)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_cluster', type=int, default=0)
    parser.add_argument('--key', type=str, default='api.key')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ensemble_model', type=str, default=None)
    parser.add_argument('--method', type=str, default="standard")
    parser.add_argument('--dev_path', type=str, default="standard")



    args = parser.parse_args()
    print(args)

    if args.debug:
        pdb.set_trace()

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    '''
    load data
    '''
    # save_dir = f'./output/{args.dataset}/{args.split}/{args.output}'
    # load ensemble weights
    save_dir = args.output

    all_lambdas = np.load(f'{save_dir}/cluster.npy') 
    if args.method == 'standard':
        all_lambdas = np.load(f'{save_dir}/cluster.npy') 
    elif args.method == "dev" or args.method == "dev_permutation":
        dev_df = pd.read_csv(args.dev_path)
        # bp()
        lambda_ex_dev = np.array(dev_df['acc'].tolist())
        lambda_ex_dev = softmax(lambda_ex_dev/0.3).reshape(1, -1)
        all_lambdas = np.repeat(lambda_ex_dev, all_lambdas.shape[0], axis=0)
    elif args.method == "cached_prior":
        all_lambdas = np.zeros((all_lambdas.shape[0], all_lambdas.shape[1]))
    rank = np.argsort(all_lambdas, axis=-1)
    '''
    def topk_by_sort(input, k, axis=None, ascending=True):
        if not ascending:
            input *= -1
        ind = np.argsort(input, axis=axis)
        ind = np.take(ind, np.arange(k), axis=axis)
        if not ascending:
            input *= -1
        val = np.take_along_axis(input, ind, axis=axis) 
        return ind, val
    from collections import Counter
    all_topk = []
    for i in range(all_lambdas.shape[0]):
        ind, val = topk_by_sort(all_lambdas[i, :], 5, ascending=False)
        all_topk.extend(ind.tolist())
    '''
    all_predictions_list = []
    # for model_name_last in os.listdir(save_dir):
    for model_name_last in range(args.num_cluster):
        # if model_name_last in ["cluster.npy", "predictions_list.jsonl", "accs", "preds", "old", "score.csv", "score_pmi.csv", "ensemble"]:
        #     continue
        # bp()
        pred_path = f'{save_dir}/{model_name_last}/predictions_list.jsonl'
        with open(pred_path, 'r') as f:
            predictions_list = json.loads(f.read())
        all_predictions_list.append(predictions_list)
    # bp()
    # my_prediction_list = all_predictions_list[7]
    '''
    ensemble
    '''
    final_predictions_list = []
    all_topk_index = []
    cluster2_final_predictions_list = defaultdict(list)
    cluster2acc = [0]*args.num_cluster
    for i in range(all_lambdas.shape[0]): # i refer to the example number
        if args.method == "cached_prior":
            # print("cluster2acc: ", cluster2acc)
            lambda_ex = np.array(cluster2acc)
            lambda_ex = softmax(lambda_ex/0.3)
            # print("lambda_ex: ", lambda_ex)
            # bp()
        else:
            lambda_ex = all_lambdas[i, :]
        lambda_ex, topk_index = zero_out(lambda_ex, args.topk)
        # bp()
        all_topk_index.extend(topk_index.tolist())
        cond_ce = [0]*len(all_predictions_list[0][0]["cond_ce"])
        domain_cond_ce = [0]*len(all_predictions_list[0][0]["domain_cond_ce"])
        uncond_ce = [0]*len(all_predictions_list[0][0]["uncond_ce"])
        # bp()
        assert len(all_predictions_list[0]) == all_lambdas.shape[0]
        assert len(all_predictions_list) == all_lambdas.shape[1]

        coeffs = torch.log(torch.FloatTensor(lambda_ex)).reshape(-1, 1)
        # bp()
        coeffs = coeffs.repeat(1, len(cond_ce))
        # coeffs = torch.log(torch.FloatTensor(lambda_ex))
        cond_ce_prob, domain_cond_ce_prob, uncond_ce_prob = [], [], []
        dcpmi_prob, pmi_prob = [], []
        lm_majority_vote_pred, dcpmi_majority_vote_pred = [0]*len(all_predictions_list[0][0]["domain_cond_ce"]),  [0]*len(all_predictions_list[0][0]["domain_cond_ce"])

        for k in range(len(lambda_ex)):
            cond_ce = all_predictions_list[k][i]["cond_ce"]
            domain_cond_ce = all_predictions_list[k][i]["domain_cond_ce"]
            cond_ce_prob.append(cond_ce)
            domain_cond_ce_prob.append(domain_cond_ce)
            uncond_ce_prob.append(all_predictions_list[k][i]["uncond_ce"])
            # majority vote
            lm_majority_vote_pred[all_predictions_list[k][i]["lm"]] += 1
            dcpmi_majority_vote_pred[all_predictions_list[k][i]["dcpmi"]] += 1

            # each pmi
            dcpmi = [ce_1 - ce_0 for ce_0, ce_1 in zip(domain_cond_ce, cond_ce)]
            pmi = [ce_1 - ce_0 for ce_0, ce_1 in zip(uncond_ce, cond_ce)]
            dcpmi_prob.append(dcpmi)
            pmi_prob.append(pmi)

            # cached prior
            if args.method == "cached_prior":
                lm_pred = cond_ce_prob.index(max(cond_ce_prob))
                lm_domain_cond_pred = domain_cond_ce_prob.index(max(domain_cond_ce_prob))
                dcpmi_pred = dcpmi.index(max(dcpmi))
                pmi_pred = pmi.index(max(pmi))
                dcpmi_prob_pred_each = dcpmi_prob.index(max(dcpmi_prob))
                pmi_prob_pred_each = pmi_prob.index(max(pmi_prob))

                pred = {
                'lm': lm_pred,
                'dcpmi' : dcpmi_pred,
                'pmi': pmi_pred,
                'dcpmi_prob_pred_each': dcpmi_prob_pred_each,
                'pmi_prob_pred_each': pmi_prob_pred_each,
                'domain_cond': lm_domain_cond_pred,
                'cond_ce': cond_ce_prob,
                'domain_cond_ce': domain_cond_ce_prob,
                'uncond_ce': uncond_ce,
                'label': all_predictions_list[0][i]['label'],
                'lm_majority_vote_pred': lm_majority_vote_pred,
                'dcpmi_majority_vote_pred': dcpmi_majority_vote_pred,
                "lambda_ex": lambda_ex.tolist()
                }
                # print(k)
                cluster2_final_predictions_list[k].append(pred)
                accs, preds, predictions_list = compute_acc(cluster2_final_predictions_list[k])
                cluster2acc[k] = accs["lm"]
        # bp() 
                # bp()



        # bp()
        cond_ce_prob = -torch.FloatTensor(cond_ce_prob)
        domain_cond_ce_prob = -torch.FloatTensor(domain_cond_ce_prob)
        uncond_ce_prob = -torch.FloatTensor(uncond_ce_prob)
        dcpmi_prob = -torch.FloatTensor(dcpmi_prob)
        pmi_prob = -torch.FloatTensor(pmi_prob)

        if args.softmax:
            # bp()
            cond_ce_prob = torch.log_softmax(cond_ce_prob, dim=-1)
            # cond_ce_prob = torch.softmax(cond_ce_prob, dim=-1)
            domain_cond_ce_prob = torch.log_softmax(domain_cond_ce_prob, dim=-1)
            uncond_ce_prob = torch.log_softmax(uncond_ce_prob, dim=-1)
            dcpmi_prob = torch.log_softmax(dcpmi_prob, dim=-1)
            pmi_prob = torch.log_softmax(pmi_prob, dim=-1)
            
        if args.coeffs == 1:
            lambda_val =  coeffs # coeffs
        else:
            lambda_val = 0
        cond_ce_prob = torch.logsumexp(cond_ce_prob + lambda_val, dim=0)
        domain_cond_ce_prob = torch.logsumexp(domain_cond_ce_prob + lambda_val, dim=0)
        uncond_ce_prob = torch.logsumexp(uncond_ce_prob + lambda_val, dim=0)
        dcpmi_prob = torch.logsumexp(dcpmi_prob + lambda_val, dim=0)
        pmi_prob = torch.logsumexp(pmi_prob + lambda_val, dim=0)

        cond_ce_prob = cond_ce_prob.tolist()
        domain_cond_ce_prob = domain_cond_ce_prob.tolist()
        uncond_ce_prob = uncond_ce_prob.tolist()
        dcpmi_prob = dcpmi_prob.tolist()
        pmi_prob = pmi_prob.tolist()

        dcpmi = [ce_1 - ce_0 for ce_0,ce_1 in zip(domain_cond_ce_prob, cond_ce_prob)]
        pmi = [ce_1 - ce_0 for ce_0,ce_1 in zip(uncond_ce_prob, cond_ce_prob)]

 
        lm_pred = cond_ce_prob.index(max(cond_ce_prob))
        lm_domain_cond_pred = domain_cond_ce_prob.index(max(domain_cond_ce_prob))
        dcpmi_pred = dcpmi.index(max(dcpmi))
        pmi_pred = pmi.index(max(pmi))
        dcpmi_prob_pred_each = dcpmi_prob.index(max(dcpmi_prob))
        pmi_prob_pred_each = pmi_prob.index(max(pmi_prob))

        lm_majority_vote_pred = lm_majority_vote_pred.index(max(lm_majority_vote_pred))
        dcpmi_majority_vote_pred = dcpmi_majority_vote_pred.index(max(dcpmi_majority_vote_pred))


        pred = {
                'lm': lm_pred,
                'dcpmi' : dcpmi_pred,
                'pmi': pmi_pred,
                'dcpmi_prob_pred_each': dcpmi_prob_pred_each,
                'pmi_prob_pred_each': pmi_prob_pred_each,
                'domain_cond': lm_domain_cond_pred,
                'cond_ce': cond_ce_prob,
                'domain_cond_ce': domain_cond_ce_prob,
                'uncond_ce': uncond_ce,
                'label': all_predictions_list[0][i]['label'],
                'lm_majority_vote_pred': lm_majority_vote_pred,
                'dcpmi_majority_vote_pred': dcpmi_majority_vote_pred,
                "lambda_ex": lambda_ex.tolist()
        }
        final_predictions_list.append(pred)
        
    all_topk_counter = Counter(all_topk_index)
    print("topk: ", all_topk_counter.most_common(10))
    accs, preds, predictions_list = compute_acc(final_predictions_list)
       # save predictions_list
    Path(f'{save_dir}/ensemble/{args.method}/top{args.topk}').mkdir(parents=True, exist_ok=True)
    predictions_list_path = f'{save_dir}/ensemble/{args.method}/top{args.topk}/predictions_list.jsonl'
    with open(predictions_list_path, 'w') as f:
        f.write(json.dumps(predictions_list))


    '''
    save path
    '''
    # print results
    print(f'{args.dataset} gets {accs}% on {args.dataset}')
