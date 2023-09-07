"""
Dataloading for EntailmentBank.
"""
from copy import deepcopy
from common import *
from proof import Proof, InvalidProofStep
import random
import json
import itertools
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
from utils import create_synthetic_seq

def read_entailmentbank_proofs(path: str, is_train: bool) -> List[Example]:
    """
    Load the EntailmentBank dataset.
    """
    data = []
    num_invalid = 0

    for count, line in enumerate(open(path)):
        ex = json.loads(line)
        hypothesis = normalize(ex["hypothesis"])
        context = extract_context(ex["context"])
        proof_text = normalize(ex["proof"].strip())
        try:
            proof = Proof(
                context,
                hypothesis,
                proof_text,
                count,
                strict=is_train,
                requires_complete=is_train,
            )
            data.append({"proof": proof, "context": context})
        except InvalidProofStep:
            assert is_train
            num_invalid += 1

    print(f"{len(data)} proofs loaded. {num_invalid} invalid ones removed.")
    return data


def collect_proved_subtrees(tree: TreeNode, prob: float) -> Iterable[TreeNode]:
    if tree.is_leaf():
        return []
    elif random.random() < prob:
        return [tree]
    else:
        return itertools.chain.from_iterable(
            collect_proved_subtrees(child, prob) for child in tree.children
        )


class StepwiseDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        path: str,
        model_name: str,
        max_input_len: int,
        max_output_len: int,
        sample_goal: str,
        subtree_proved_prob: float,
        subtree_proved_all_or_none: bool,
        is_train: bool,
    ) -> None:
        super().__init__()
        max_len = max(max_input_len, max_output_len)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length=max_len
        )

        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.sample_goal = sample_goal
        self.subtree_proved_prob = subtree_proved_prob
        self.subtree_proved_all_or_none = subtree_proved_all_or_none
        self.is_train = is_train
        self.data = read_entailmentbank_proofs(path, is_train)
        if not self.is_train:
            print(self.data[0])
            self.data = self.get_example_eval(self.data)
            print(self.data[0])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        if self.is_train:
            ex = self.data[idx]
            return self.get_example_train(ex)
        else:
            return self.data[idx]

    def collate(self, examples: List[Example]) -> Batch:
        ex = examples[0]
        inp = ["$premises$: " + ex["input_fake_seq"][i] + " $conclusion$: " for i in range(len(ex["input_fake_seq"]))]
        return {
            "input_seq": ex["input_seq"],
            "input_fake_seq": ex["input_fake_seq"],
            "input_fake_ident": ex["input_fake_ident"],
            "input_fake_seq_convert": inp,
            "output_seq": ex["output_seq"],
            "hypothesis": ex["proof"].hypothesis,
            "hypothesis_sample": ex["hypothesis_sample"],
            "context": ex["context"],
            "proof_gt": ex["proof"].proof_text,
            "partial_proof": ex["partial_proof"],
            "stepwise_goal": ex["output_seq"],
            "proof_id": ex["proof"].proof_id,
        }

    def get_example_train(self, ex: Example) -> Example:
        proof = ex["proof"].shuffle_context()

        tree = proof.to_tree()
        int_node = random.choice(get_internal_nodes(tree))

        if self.sample_goal == "hypothesis":
            goal_node = tree.get_tree_root()
        else:
            assert self.sample_goal == "intermediates"
            ancestors = int_node.get_ancestors()
            assert int_node not in ancestors
            ancestors.append(int_node)
            goal_node = random.choice(ancestors)

        proved_subtrees = [node for node in int_node.children if not node.is_leaf()]
        if int_node is not goal_node:
            unproved_child = int_node
            for node in int_node.iter_ancestors():
                for child in node.children:
                    if child is unproved_child or child.is_leaf():
                        continue
                    proved_subtrees.extend(collect_proved_subtrees(child, self.subtree_proved_prob))
                if node is goal_node:
                    break
                else:
                    unproved_child = node
        proved_subtrees.reverse()
        random.shuffle(proved_subtrees)
        partial_proof = " ".join(serialize(t) for t in proved_subtrees)


        input_seq = f"$hypothesis$ = {goal_node.sent} ; $context$ = {proof.serialize_context()} ; $proof$ = {partial_proof}"

        premises = [node.name for node in int_node.children]
        random.shuffle(premises)
        output_seq = " & ".join(premises)
        if goal_node is int_node:
            output_seq = output_seq + " -> hypothesis;"
        else:
            output_seq = output_seq + f" -> int: {int_node.sent};"

        ex = deepcopy(ex)
        ex["proof"] = proof
        ex["input_seq"] = input_seq
        ex["output_seq"] = output_seq
        return ex

    def get_example_eval(self, examples: List[Example]) -> List[Example]:
        eval_data = []
        for ex in examples:
            proof = ex["proof"]
            context = ex["context"]
            tree = proof.to_tree()
            for i, int_node in enumerate(get_internal_nodes(tree)):
                ancestors = int_node.get_ancestors()
                assert int_node not in ancestors
                ancestors.append(int_node)
                for goal_node in ancestors:
                    proved_subtrees = [node for node in int_node.children if not node.is_leaf()]
                    if int_node is not goal_node:
                        unproved_child = int_node
                        for node in int_node.iter_ancestors():
                            for child in node.children:
                                if child is unproved_child or child.is_leaf():
                                    continue
                                proved_subtrees.extend(collect_proved_subtrees(child, self.subtree_proved_prob))
                            if node is goal_node:
                                break
                            else:
                                unproved_child = node
                    proved_subtrees.reverse()
                    partial_proof = " ".join(serialize(t) for t in proved_subtrees)

                    input_seq = f"$hypothesis$ = {goal_node.sent} ; $context$ = {proof.serialize_context()} ; $proof$ = {partial_proof}"
                    premises = [node.name for node in int_node.children]
                    output_seq = " & ".join(premises)
                    if goal_node is int_node:
                        output_seq = output_seq + " -> hypothesis;"
                    else:
                        output_seq = output_seq + f" -> int: {int_node.sent};"
                    ex_new = deepcopy(ex)
                    ex_new["input_seq"] = input_seq
                    ex_new["output_seq"] = output_seq
                    ex_new["partial_proof"] = partial_proof
                    # should be fixed
                    ex_new["hypothesis_sample"] = goal_node.sent 
                    # get correct candidates but with redudant steps
                    int_node_tmp = deepcopy(int_node)
                    ex_new["input_fake_seq"], ex_new["input_fake_ident"] = create_synthetic_seq(int_node_tmp, context)
                    #ex_new["input_fake_seq"], ex_new["input_fake_ident"] = "none", "none"

                    eval_data.append(ex_new)
        return eval_data


class ProofDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        stepwise: bool,
        sample_goal: str,
        model_name: str,
        max_input_len: int,
        max_output_len: int,
        batch_size: int,
        num_workers: int,
        path_train: str,
        path_val: str,
        path_test: str,
        subtree_proved_prob: float,
        subtree_proved_all_or_none: bool,
    ) -> None:
        super().__init__()
        assert dataset in ("entailmentbank", "ruletaker")
        self.dataset = dataset
        self.stepwise = stepwise
        self.sample_goal = sample_goal
        self.model_name = model_name
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.path_train = path_train
        self.path_val = path_val
        self.path_test = path_test
        self.subtree_proved_prob = subtree_proved_prob
        self.subtree_proved_all_or_none = subtree_proved_all_or_none

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            if self.stepwise:
                self.ds_train = StepwiseDataset(
                    self.dataset,
                    self.path_train,
                    self.model_name,
                    self.max_input_len,
                    self.max_output_len,
                    self.sample_goal,
                    self.subtree_proved_prob,
                    self.subtree_proved_all_or_none,
                    is_train=True,
                )

        if stage in (None, "fit", "validate"):
            if self.stepwise:
                self.ds_val = StepwiseDataset(
                    self.dataset,
                    self.path_val,
                    self.model_name,
                    self.max_input_len,
                    self.max_output_len,
                    self.sample_goal,
                    self.subtree_proved_prob,
                    self.subtree_proved_all_or_none,
                    is_train=False,
                )

        if stage in (None, "test"):
            if self.stepwise:
                self.ds_test = StepwiseDataset(
                    self.dataset,
                    self.path_test,
                    self.model_name,
                    self.max_input_len,
                    self.max_output_len,
                    self.sample_goal,
                    self.subtree_proved_prob,
                    self.subtree_proved_all_or_none,
                    is_train=False,
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.ds_train.collate,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.ds_val.collate,
            pin_memory=True,
            drop_last=False,
        )


if __name__ == "__main__":
    path_train = "/home/ysuay/codes/LLM+Reason/codes/NLProofS/data/entailment_trees_emnlp2021_data_v3/dataset/task_2/train.jsonl"
    path_val = "/home/ysuay/codes/LLM+Reason/codes/NLProofS/data/entailment_trees_emnlp2021_data_v3/dataset/task_2/dev.jsonl"
    print(path_train)

    ds_train = StepwiseDataset(
        path=path_train,
        model_name="t5-large",
        dataset="entailmentbank",
        max_input_len=1024,
        max_output_len=128,
        sample_goal="intermediates",
        subtree_proved_prob=0.75,
        subtree_proved_all_or_none=0.75,        
        is_train=False)
    print(len(ds_train))
    #print(ds_train[0])
    ret = ds_train.collate([ds_train[0], ds_train[1]])
    print(ret)

