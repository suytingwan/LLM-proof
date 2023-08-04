import sys
sys.path.append('../')

from common import *
from copy import deepcopy
import itertools
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import json
import random
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi


def sample_similar_sentence(query: str, corpus: List[str]) -> str:
    # Sample a sentence in `corpus` that is similar to `query`
    assert query not in corpus
    tokenized_query = query.split()
    tokenized_corpus = [sent.split() for sent in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = F.softmax(torch.tensor(bm25.get_scores(tokenized_query)), dim=0)
    dist = Categorical(probs=scores)
    sent = corpus[dist.sample()]
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


def powerset(iterable: Iterable[Any]) -> List[Tuple[Any]]:
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(
        itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    )


class EntailmentDataset(Dataset):
    def __init__(
        self,
        path: str,
        model_name: str,
        max_num_premises: int,
        split: str,
        max_input_len: int,
        irrelevant_distractors_only: bool,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length=max_input_len
        )
        assert split in ("train", "val")
        self.split = split
        self.max_num_premises = max_num_premises
        self.max_input_len = max_input_len
        self.irrelevant_distractors_only = irrelevant_distractors_only
        self.data = self.preprocess(path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        ex = self.data[idx]
        premises = deepcopy(ex["premises"])
        random.shuffle(premises)
        premises = ". ".join(premises) + "."
        return {
            "premises": premises,
            "conclusion": ex["conclusion"],
            "label": ex["label"],
        }

    def preprocess(self, path:str) -> List[Example]:
        raise NotImplementedErrot

    def collate(self, examples: List[Example]) -> Batch:
        entailment = self.tokenizer(
            [(ex["premises"], ex["conclusion"]) for ex in examples],
            padding="longest",
            truncation="longest_first",
            max_length=self.max_input_len,
            return_tensors="pt",
        )
        label = torch.tensor([ex["label"] for ex in examples], dtype=torch.int64)
        return {
            "premises": [ex["premises"] for ex in examples],
            "conclusion": [ex["conclusion"] for ex in examples],
            "input_ids": entailment["input_ids"],
            "attention_mask": entailment["attention_mask"],
            "label": label,
        }


class EntailmentBankDataset(EntailmentDataset):
    def preprocess(self, path: str) -> List[Example]:
        """
        Extract positive and negative examples from ground truth proof trees.
        """
        data = []
        num_pos = 0
        num_neg = 0

        for line in open(path):
            ex = json.loads(line)
            context = extract_context(ex["context"])
            pos, neg = self.extract_examples(ex, context)
            data.extend(pos)
            data.extend(neg)
            num_pos += len(pos)
            num_neg += len(neg)

        random.shuffle(data)
        print(f"#positives: {num_pos}\n#pseudo-negatives: {num_neg}")

        return data

    def extract_examples(
        self, ex: Example, context: OrderedDict[str, str]
    ) -> Tuple[List[Example], List[Example]]:
        """
        Extract positive and negative examples from a proof tree.
        """
        positives = []
        negatives = []
        tree = deserialize(ex["hypothesis"], context, ex["proof"])

        def create_positive(premises: List[str], conclusion: str) -> None:
            assert len(premises) >= 2
            positives.append(
                {"premises": premises, "conclusion": conclusion, "label": True}
            )
        def create_negative(premises: List[str], conclusion: str) -> None:
            assert len(premises) >= 2
            negatives.append(
                {"premises": premises, "conclusion": conclusion, "label": False}
            )

        for node in tree.traverse():
            if node.is_leaf():
                continue

            if self.split != "train":
                premises = [child.sent for child in node.children]
                if len(premises) >= 2:
                    create_positive(premises, node.sent)
            else:
                # 1. Enumerate all combinations of premises leading to node.sent.
                for premise_nodes in enumerate_premise_nodes(
                    node, self.max_num_premises
                ):
                    premises = [pn.sent for pn in premise_nodes]
                    num_premises = len(premises)

                    if num_premises >= 2:
                        create_positive(premises, node.sent)

                        # 2. Perturbe them to generate negatives.
                        for i, p in enumerate(premises):
                            if self.irrelevant_distractors_only:
                                candidates = [
                                    sent
                                    for sent in context.values()
                                    if sent not in premises
                                ]
                            else:
                                candidates = [
                                    sent for sent in context.values() if sent != p
                                ]
                            alternative = sample_similar_sentence(p, candidates) # find a most similar but uncorrect substitute premise
                            prems = deepcopy(premises)
                            prems[i] = alternative
                            create_negative(prems, node.sent)

                        if num_premises > 2:
                            for subset in powerset(premises):
                                if 2 <= len(subset) < num_premises: # subset exclude the entire set
                                    create_negative(list(subset), node.sent)
            #import pdb
            #pdb.set_trace()

        if self.split == "train":
            leaf_sents = [node.sent for node in tree.get_leaves()]
            for s1 in leaf_sents:
                for s2 in leaf_sents:
                    if s1 == s2:
                        continue
                    create_negative([s1, s2], s1)
        return positives, negatives

 
class RuleTakerDataset(EntailmentDataset):
    def preprocess(self, path: str) -> List[Example]:
        """
        Extract positive and negative examples from ground truth proof trees.
        """
        data = []

        for line in open(path):
            ex = json.loads(line)
            pos, neg = self.extract_examples(ex)
            data.extend(pos)
            data.extend(neg)

        data = list(set(data))
        random.shuffle(data)
        data = [
            {"premises": list(ex[0]), "conclusion": ex[1], "label": ex[2]}
            for ex in data
        ]
        num_pos = sum([1 for x in data if x["label"] == True])
        num_neg = sum([1 for x in data if x["label"] == False])
        print(f"#positives: {num_pos}\n#pseudo-negatives: {num_neg}")

        return data

    def extract_examples(
        self,
        ex: Example,
    ) -> Tuple[List[Any], List[Any]]:
        """
        Extract positive and negative examples from a proof tree.
        """
        context = extract_context(ex["context"])
        positives = []
        negatives = []

        def create_positive(premises: List[str], conclusion: str) -> None:
            positives.append((tuple(premises), conclusion, True))

        def create_negative(premises: List[str], conclusion: str) -> None:
            negatives.append((tuple(premises), conclusion, False))

        for proof in ex["proofs"]:
            tree = deserialize(ex["hypothesis"], context, proof)
            if tree is None:
                assert proof == ""
                continue

            for node in tree.traverse():
                if node.is_leaf():
                    continue

                premises = [child.sent for child in node.children]
                if ex["answer"] == False and node.is_root():
                    create_negative(premises, node.sent)
                    continue
                else:
                    create_positive(premises, node.sent)

                if self.split == "train":
                    if "does not " in node.sent:
                        create_negative(premises, node.sent.replace("does not ", ""))
                    elif "do not " in node.sent:
                        create_negative(premises, node.sent.replace("do not", ""))
                    elif "cannot " in node.sent:
                        create_negative(premises, node.sent.replace("cannot", "can"))
                    elif "not " in node.sent:
                        create_negative(premises, node.sent.replace("not ", ""))

                    if node.sent.startswith("i don't think "):
                        create_negative(
                            premises, node.sent.replace("i don't think ", "")
                        )
                    else:
                        create_negative(premises, f"i don't think {node.sent}")

                    for i, p in enumerate(premises):
                        if self.irrelevant_distractors_only:
                            candidates = [
                                sent
                                for sent in context.values()
                                if sent not in premises
                            ]
                        else:
                            candidates = [
                                sent for sent in context.values() if sent != p
                            ]
                        if len(candidates) == 0:
                            continue
                        alternative = sample_similar_sentence(p, candidates)
                        prems = deepcopy(premises)
                        prems[i] = alternative
                        create_negative(prems, node.sent)
        return positives, negatives


if __name__ == "__main__":

    #path_train = "/home/ysuay/codes/LLM+Reason/codes/NLProofS/data/entailment_trees_emnlp2021_data_v3/dataset/task_2/train.jsonl"
    #path_val = "/home/ysuay/codes/LLM+Reason/codes/NLProofS/data/entailment_trees_emnlp2021_data_v3/dataset/task_2/dev.jsonl"

    #ds_train = EntailmentBankDataset(
    #               path=path_train,
    #               model_name="roberta-large",
    #               max_num_premises=4,
    #               split="train",
    #               max_input_len=256,
    #               irrelevant_distractors_only=False)

    #fw = open('entailment_verifier_task2_samples.txt', 'w')
    #for sample in ds_train:
    #    fw.write(json.dumps(sample)+'\n')
    #fw.close()

    path_train = "/home/ysuay/codes/LLM+Reason/codes/NLProofS/data/entailment_trees_emnlp2021_data_v3/dataset/task_1/train.jsonl"
    path_val = "/home/ysuay/codes/LLM+Reason/codes/NLProofS/data/entailment_trees_emnlp2021_data_v3/dataset/task_1/dev.jsonl"

    ds_train = EntailmentBankDataset(
                   path=path_train,
                   model_name="roberta-large",
                   max_num_premises=4,
                   split="train",
                   max_input_len=256,
                   irrelevant_distractors_only=False)

    fw = open('entailment_verifier_task1_samples.txt', 'w')
    for sample in ds_train:
        fw.write(json.dumps(sample)+'\n')
    fw.close()

   #path_train = "/home/ysuay/codes/LLM+Reason/codes/NLProofS/data/proofwriter-dataset-V2020.12.3/preprocessed_OWA/depth-3ext/meta-train.jsonl"
    #path_val = "/home/ysuay/codes/LLM+Reason/codes/NLProofS/data/proofwriter-dataset-V2020.12.3/preprocessed_OWA/depth-3ext/meta-dev.jsonl"

    #ds_train = RuleTakerDataset(
    #               path=path_train,
    #               model_name="roberta-large",
    #               max_num_premises=4,
    #               split="train",
    #               max_input_len=128,
    #               irrelevant_distractors_only=True)

    #fw = open('ruletaker_verifier_samples.txt', 'w')
    #for sample in ds_train:
    #    fw.write(json.dumps(sample)+'\n')
    #fw.close()

    print(ds_train[0])
