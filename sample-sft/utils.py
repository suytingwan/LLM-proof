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

def sample_redundant_sentence(query: str, corpus: List[str]) -> str:
    assert query not in corpus
    sent = random.sample(corpus, 1)
    return sent

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

def create_synthetic_seq(int_node, context):
    premises = [node.sent for node in int_node.children]
    identities = [node.name for node in int_node.children]
    input_fake_seq = []
    input_fake_ident = []

    candidate_keys = {}
    for key in context.keys():
        candidate_keys[context[key]] = key

    for i, p in enumerate(premises):
        # irrelevant distractors only
        candidates = []
        candidates = [sent for sent in context.values() if sent not in premises]

        # 1. substitute similar node in int_node
        alternative = sample_similar_sentence(p, candidates)
        prems = deepcopy(premises)
        prems[i] = alternative
        idents = deepcopy(identities)
        idents[i] = candidate_keys[alternative]
        # 2. substitute different node in int_node
        alternative2 = sample_unsimilar_sentence(p, candidates)
        prems2 = deepcopy(premises)
        prems2[i] = alternative2
        idents2 = deepcopy(identities)
        idents2[i] = candidate_keys[alternative2]
        # 3. add redudant node in int_node
        alternative3 = sample_redundant_sentence(p, candidates)
        prems3 = deepcopy(premises)
        prems3.append(alternative3[0])
        idents3 = deepcopy(identities)
        idents3.append(candidate_keys[alternative3[0]])

        input_fake_seq.append(". ".join(prems))
        input_fake_seq.append(". ".join(prems2))
        input_fake_seq.append(". ".join(prems3))

        input_fake_ident.append(" & ".join(idents))
        input_fake_ident.append(" & ".join(idents2))
        input_fake_ident.append(" & ".join(idents3))  
    return input_fake_seq, input_fake_ident
