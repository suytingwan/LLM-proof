from copy import deepcopy
import itertools
import torch
import torch.nn.functional as F
import json
import random
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class EntailmentBankDataset(Dataset):
    def __init__(
        self,
        path: str,
        max_num_premises: int,
        split: str,
    ) -> None:
        super().__init__()
        assert split in ("train", "val")
        self.split = split
        self.max_num_premises = max_num_premises
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
        tree = deserialize(ex["hypothesis"], context, ex["proof"])

        def create_positive(premises: List[str], conclusion: str) -> None:
            assert len(premises) >= 2
            positives.append(
                {"premises": premises, "conclusion": conclusion, "label": True}
            )

        for node in tree.traverse():
            if node.is_leaf():
                continue

            if self.split == "train":
                for premise_nodes in enumerate_premise_nodes(
                    node, self.max_num_premises
                ):
                    premises = [pn.sent for pn in premise_nodes]
                    num_premises = len(premises)

                    if num_premises >= 2:
                        create_positive(premises, node.sent)

        return positives


if __name__ == "__main__":
    path_train = "../data/entailment_trees_emnlp2021_data_v3/dataset/task_1/train.jsonl"
    ds_train = EntailmentBankDataset(
                   path=path_train,
                   max_num_premises=4,
                   split="train")

    fw = open('reasoner_task1_samples.txt', 'w')
    for sample in ds_train:
        fw.write(json.dumps(sample)+'\n')
    fw.close()

