# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

import torch
from metaseq import utils
import pickle
from tqdm import trange
import numpy as np
from collections import defaultdict
from copy import deepcopy
from metaseq.scripts.train_clusterer import NumberNormalizingVectorizer


def print_r0(x, file=None):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(x, file=file)

class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "metaseq.scripts.train_clusterer"
        return super().find_class(module, name)


def load_model(path_to_model):
    with open(path_to_model, 'rb') as f:
        unpickler = MyCustomUnpickler(f)
        out = unpickler.load()
    return out


def _average(models, weights=None):
    state_dicts = [model.state_dict() for model in models]
    with torch.no_grad():
        merged = {}
        
        for key in state_dicts[0]:
            merged[key] = torch.sum(torch.stack([sd[key] * weight for sd, weight in zip(state_dicts, weights)]), axis=0)
        return merged

def average(models, weights=None):
    res = deepcopy(models[0])
    averaged_model = _average(models, weights=weights)
    res.load_state_dict(averaged_model)
    return res

def _get_context_clusters(tokenizer, tfidf, kmeans, net_input, idx, random_clusters=False):
    """
    get clusters for every context.
    this is pretty slow! we're working on speeding this up.
    """
    id_subsequence = net_input[:,:idx].tolist()
    results = defaultdict(list)
    output = []
    for i in range(net_input.shape[0]):
        document_subsequences = []
        if random_clusters:
            for j in range(idx):
                distances = torch.FloatTensor(np.random.random(torch.distributed.get_world_size()))
                results[i].append(distances.unsqueeze(0))
        else:
            for j in range(idx):
                decoded_tokens = tokenizer.decode_batch(net_input[:,:j].tolist())
                document_subsequences.extend(decoded_tokens)
            _, distances = kmeans.predict(torch.from_numpy(tfidf.transform(document_subsequences)),
                                        return_distances=True)
            results[i].append(distances)
        for j in range(idx, net_input.shape[1]):
            results[i].append(distances[-1, :].unsqueeze(0))
        output.append(torch.cat(results[i], 0).t().unsqueeze(0))
    output = torch.cat(output, 0)
    return output


class SequenceScorerBTM(object):
    """Scores the target for a given source sentence."""

    def __init__(
        self,
        tgt_dict,
        path_to_clusterer=None,
        random_clusters=False,
        softmax_batch=None,
        compute_alignment=False,
        num_clusters=1,
        eos=None,
        symbols_to_strip_from_output=None,
        temperature=0.1,
        cluster_ratio=1.0,
        average=False,
        tokenizer=None,
        ensemble_type='product_of_experts',
    ):
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.softmax_batch = softmax_batch or sys.maxsize
        self.temperature = temperature
        self.cluster_ratio = cluster_ratio
        self.tokenizer = tokenizer
        self.average = average
        self.random_clusters = random_clusters
        self.ensemble_type = ensemble_type
        if num_clusters > 1 and not self.random_clusters:
            self.kmeans = load_model(f"{path_to_clusterer}/kmeans.pkl")
            
            self.tfidf = load_model(f"{path_to_clusterer}/tfidf.pkl")
            # self.kmeans = load_model(f"/private/home/suching/metaseq-internal/s2orc_clusters/8/kmeans.pkl")
            # self.tfidf = load_model(f"/private/home/suching/metaseq-internal/s2orc_clusters/8/tfidf.pkl")
        else:
            self.kmeans = None
            self.tfidf = None
        assert self.softmax_batch > 0
        self.compute_alignment = compute_alignment
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None
            else {self.eos}
        )

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        net_input = sample["net_input"]
        start_idx = int(net_input.shape[-1] * self.cluster_ratio)
        len_models = torch.distributed.get_world_size() if not self.average else len(models)
        

        if len_models > 1 and (self.kmeans or self.random_clusters) and self.ensemble_type in ['clustering', 'product_of_experts']:
            cluster_distances  = _get_context_clusters(self.tokenizer,
                                                     self.tfidf,
                                                     self.kmeans,
                                                     net_input,
                                                     start_idx,
                                                     random_clusters=self.random_clusters)
            cluster_distances = torch.nn.functional.softmax(-cluster_distances**2 / self.temperature, dim=1)
            if kwargs['topk'] > 0:
                prior_inds = sorted(sorted(range(len(kwargs['prior'])), key=lambda i: kwargs['prior'][i])[-kwargs['topk']:])
                kwargs['prior'] = [kwargs['prior'][i] if i in prior_inds else 0 for i in range(len(kwargs['prior']))]
                kwargs['prior'] = [p/sum(kwargs['prior']) for p in kwargs['prior']]
        else:
            cluster_distances = None
        
        if self.average:
            # models = utils.move_to_cpu(models)
            # TODO: make this not cheat with the prior
            models = [average(models, weights=kwargs['prior'].mean(-1).squeeze(0).tolist())]
            len_models = 1
        def batch_for_softmax(dec_out, target):
            # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
            first, rest = dec_out[0], dec_out[1:]
            bsz, tsz, dim = first.shape
            if bsz * tsz < self.softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        orig_target = sample["target"]

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        model_probs = []
        for model in models:
            
            if not kwargs.get('decoder_out'):
                decoder_out = model(net_input)
            else:
                decoder_out = kwargs.get('decoder_out')
            attn = decoder_out[1] if len(decoder_out) > 1 else None
            if type(attn) is dict:
                attn = attn.get("attn", None)

            batched = batch_for_softmax(decoder_out, orig_target)
            probs, idx = None, 0
            for bd, tgt, is_single in batched:
                sample["target"] = tgt
                curr_prob = model.get_normalized_probs(
                    bd, log_probs=not kwargs.get('ensemble') and len_models == 1, sample=sample
                ).data
                if is_single:
                    probs = gather_target_probs(curr_prob, orig_target)
                else:
                    if probs is None:
                        probs = curr_prob.new(orig_target.numel())
                    step = curr_prob.size(0) * curr_prob.size(1)
                    end = step + idx
                    tgt_probs = gather_target_probs(
                        curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt
                    )
                    probs[idx:end] = tgt_probs.view(-1)
                    idx = end
                sample["target"] = orig_target

            probs = probs.view(sample["target"].shape)

            if kwargs.get('ensemble') and len_models > 1:
                model_probs.append(probs.unsqueeze(0))
            else:
                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs.add_(probs)
                if attn is not None:
                    if torch.is_tensor(attn):
                        attn = attn.data
                    else:
                        attn = attn[0]
                    if avg_attn is None:
                        avg_attn = attn
                    else:
                        avg_attn.add_(attn)
        if len_models > 1 or kwargs.get('ensemble'):
            if kwargs.get('ensemble') and len_models > 1:
                if kwargs.get('all_reduce'):
                    gather_probs = [torch.ones_like(model_probs[0]).cuda() for _ in range(len_models)]
                    torch.distributed.all_gather(gather_probs, model_probs[0])
                    model_probs = torch.cat(gather_probs, dim=0)
                else:
                    model_probs = torch.cat(model_probs, dim=0)
                ## simple averaging
                if kwargs.get('ensemble_average'):
                    avg_probs = torch.mean(model_probs, dim=0)
                    weights = torch.tensor([ 1 / len_models]).repeat(
                                    len_models, model_probs.shape[1], model_probs.shape[2]).to(model_probs)
                elif kwargs.get('ensemble_weighted_average'):
                    # weighted averaging
                    if self.ensemble_type == 'product_of_experts':
                        # get t-1 probabilities
                        weights = model_probs[:, :, :-1].clone()
                        # get t-1 cluster distances
                        cluster_distances = cluster_distances[:, :, :-1]
                        cluster_distances = cluster_distances.to(weights).transpose(0,1)
                        # compute product of experts
                        denom = (weights.clone() * cluster_distances).sum(0)
                        weights = weights * cluster_distances / denom

                    elif self.ensemble_type == 'clustering':
                        # use cluster distances directly
                        weights = cluster_distances.to(model_probs).transpose(0,1)

                    elif self.ensemble_type == 'bayes':
                        # get t-1 probabilities
                        weights = model_probs[:, :, :-1].clone()
                        # setup the prior
                        if kwargs.get('prior') is not None:
                            if len_models == 1:
                                priors = torch.ones_like(weights)
                            else:
                                priors = kwargs['prior']
                        else:
                            # uniform
                            priors = [1 / len_models] * len_models
                            temperature = 1
                        # build denom
                        denom = weights.clone()
                        for ix, prior in enumerate(priors):
                            denom[ix, :].mul_(prior)
                        denom = denom.sum(0)
                        # bayes it!
                        for ix, prior in enumerate(priors):
                            weights[ix, :].mul_(prior).div_(denom)

                    if self.ensemble_type != 'clustering':
                        # add uniform likelihood for first token
                        beginning_weights = torch.tensor([1 / len_models]).repeat(
                                                len_models, model_probs.shape[1], 1).to(weights)
                        weights = torch.cat([beginning_weights, weights], -1)

                    if kwargs['topk'] > 0:
                        # print("weights", weights)
                        new_weights = torch.transpose(torch.squeeze(weights), 0, 1)
                        topk, indices = torch.topk(new_weights, kwargs['topk'])
                        new_weights = torch.zeros_like(new_weights).scatter_(1, indices, topk)
                        new_weights = new_weights / new_weights.sum(dim=-1, keepdim=True)
                        # print("new_weights", new_weights)
                        weights = torch.transpose(new_weights, 0, 1).unsqueeze(1)

                    avg_probs = torch.einsum("ebs,ebs->bs", (weights, model_probs))

                avg_probs.log_()
                if avg_attn is not None:
                    if kwargs['ensemble']:
                        avg_attn.div_(len_models)
                    else:
                        avg_attn.div_(len_models)

            else:
                weights = None
                avg_probs.div_(len_models)
                avg_probs.log_()
                if avg_attn is not None:
                    avg_attn.div_(len_models)
        else:
            weights = None
            avg_probs.div_(len_models)
            if avg_attn is not None:
                avg_attn.div_(len_models)
            src = avg_probs

        bsz = avg_probs.size(0)
        hypos = []
        start_idxs = sample["start_indices"] if "start_indices" in sample else [0] * bsz
        for i in range(bsz):
            # remove padding from ref
            ref = (
                utils.strip_pad(sample["target"][i, start_idxs[i] :], self.pad)
                if sample["target"] is not None
                else None
            )
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i] : start_idxs[i] + tgt_len]
            score_i = avg_probs_i.sum() / tgt_len
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                if self.compute_alignment:
                    alignment = utils.extract_hard_alignment(
                        avg_attn_i,
                        sample["net_input"]["src_tokens"][i],
                        sample["target"][i],
                        self.pad,
                        self.eos,
                    )
                else:
                    alignment = None
            else:
                avg_attn_i = alignment = None
            hypos.append(
                [
                    {
                        "tokens": ref,
                        "expert_probs": weights.mean(-1) if weights is not None else None,
                        "score": score_i,
                        "prior": kwargs['prior'],
                        "attention": avg_attn_i,
                        "alignment": alignment,
                        "positional_scores": avg_probs_i,
                    }
                ]
            )
        return hypos