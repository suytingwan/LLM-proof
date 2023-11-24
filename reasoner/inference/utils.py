from torch.distributions.categorical import Categorical
from rank_bm25 import BM25Okapi
from common import *
import random
import torch.nn.functional as F
from copy import deepcopy

def sample_similar_sentence(query: str, corpus: List[str]) -> str:
    assert query not in corpus
    tokenized_query = query.split()
    tokenized_corpus = [sent.split() for sent in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = F.softmax(torch.tensor(bm25.get_scores(tokenized_query)), dim=0)
    dist = Categorical(probs=scores)
    sent = corpus[dist.sample()]
    return sent

def sample_unsimilar_sentence(query: str, corpus: List[str]) -> str:
    assert query not in corpus
    tokenized_query = query.split()
    tokenized_corpus = [sent.split() for sent in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = F.softmax(torch.tensor(-bm25.get_scores(tokenized_query)), dim=0)
    dist = Categorical(probs=scores)
    sent = corpus[dist.sample()]
    return sent

def sample_topk_sentence(query: str, corpus: List[str]) -> List[str]:
    assert query not in corpus
    tokenized_query = query.split()
    tokenized_corpus = [sent.split() for sent in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = F.softmax(torch.tensor(bm25.get_scores(tokenized_query)), dim=0)
    values, indexs = torch.topk(scores, min(len(tokenized_corpus), 5))
    indexs = indexs.tolist()
    sents = [corpus[ind] for ind in indexs]
    return sents

def enumerate_premise_nodes(node: TreeNode, max_num: int) -> List[List[TreeNode]]:
    all_premises: List[List[TreeNode]] = [[]]

    for child in node.children:
        if child.is_leaf():
            for premises in all_premises:
                premises.append(child)
        else:
            prev_all_premises = all_premises
            all_premises = []
            for premises in prev_all_premises:
                all_premises.append(premises + [child])
            for child_premises in enumerate_premise_nodes(child, max_num - 1):
                for premises in prev_all_premises:
                    all_premises.append(premises + child_premises)
    return [premises for premises in all_premises if len(premises) <= max_num]

def create_synthetic_seq(int_node, context, seen_leaves, all_leaves=None, sample="random"):
    assert sample in ("random", "similar")
    premises = [node.sent for node in int_node.children]
    identities = [node.name for node in int_node.children]

    seen_leave_values = [context[seen_ident] for seen_ident in seen_leaves]
    input_fake_seq = []
    input_fake_ident = []

    candidate_keys = {}
    for key in context.keys():
        candidate_keys[context[key]] = key

    for i, p in enumerate(premises):
        # irrelevant distractors only
        if all_leaves is None:
            candidates = [sent for sent in context.values() if sent not in premises + seen_leave_values]
        else:
            all_leave_values = [context[leaf] for leaf in all_leaves]
            candidates = [sent for sent in all_leave_values if sent not in premises + seen_leave_values]
        # for task 1
        if len(candidates) == 0:
            input_fake_seq.append(premises)
            input_fake_ident.append(identities)
            print("catch entire tree")
            continue
        if sample == "BM25":
            alternatives = sample_topk_sentence(p, candidates)
        else:
            alternatives = random.sample(candidates, min(len(candidates), 5))

        for alternative in alternatives:
            prems = deepcopy(premises)
            prems[i] = alternative
            idents = deepcopy(identities)
            idents[i] = candidate_keys[alternative]
            input_fake_seq.append(prems)
            input_fake_ident.append(idents)
        
    return input_fake_seq, input_fake_ident

def create_synthetic_2hop_seq(int_node, context):
    premises = [node.sent for node in int_node.children]
    identities = [node.name for node in int_node.children]
    input_fake_seq = []
    input_fake_ident = []

    candidate_keys = {}
    for key in context.keys():
        candidate_keys[context[key]] = key

    for i, p in enumerate(premises):
        candidates = [sent for sent in context.values() if sent not in premises]
        if len(candidates) == 0:
            input_fake_seq.append(premises)
            input_fake_ident.append(identities)
            print("catch entire tree")
            continue
        elif len(candidates) == 1:
            alternative = candidates[0]
            prems = deepcopy(premises)
            prems[i] = alternative
            idents = deepcopy(identities)
            idents[i] = candidate_keys[alternative]
            input_fake_seq.append(prems)
            input_fake_ident.append(idents)
        else:
            alternatives = sample_random_2hop_sentence(p, candidates)
            if len(prems) < 2:
                continue
            prems = deepcopy(premises)
            prems[i] = alternatives[0]
            idents = deepcopy(identities)
            idents[i] = candidate_keys[alternative[0]]
            input_fake_seq.append(prems)
            input_fake_ident.append(idents)
            while True:
                j = random.randint(0, len(prems)-1)
                if j != i:
                    break
            prems2 = deepcopy(prems)
            prems2[i] = "TBD"
            prems2[j] = alternatives[1]
            idents2 = deepcopy(idents)
            idents2[i] = "int"
            idents2[j] = candidate_keys[alternative[1]]
            input_fake_seq.append(prems2)
            input_fake_ident.append(idents2)
    return input_fake_seq, input_fake_ident
