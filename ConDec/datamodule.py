"""
Dataloading for EntailmentBank and RuleTaker.
"""
from copy import deepcopy
from common import *
from proof import Proof, InvalidProofStep
import random
import json
import numpy as np
import itertools
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
from collections import defaultdict


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
            data.append({"proof": proof})
        except InvalidProofStep:
            assert is_train
            num_invalid += 1

    print(f"{len(data)} proofs loaded. {num_invalid} invalid ones removed.")
    return data

def read_entailmentbank_contrast_proofs(path: str, is_train: bool) -> List[Example]:
    """
    Load the EntailmentBank dataset.
    """
    data = defaultdict(dict)
    num_invalid = 0

    for count, line in enumerate(open(path)):
        ex = json.loads(line)
        hypothesis = ex["hypothesis"]
        context = ex["context"]
        proof_text = ex["proof_gt"].strip()
 
        try:
            proof = Proof(
                context,
                hypothesis,
                proof_text,
                count,
                strict=is_train,
                requires_complete=is_train,
            )
 
            key_input = f"$context$ = {proof.serialize_context()}"
            key_premise_str = ex["stepwise_goal"].split(" -> ")[0]
            key_premises = key_premise_str.split(" & ")
            key_output = " & ".join(sorted(key_premises))

            data[key_input][key_output] = {}
            data[key_input][key_output]["input_fake_ident"] = []
            data[key_input][key_output]["decode_conclusion"] = []

            for i, score in enumerate(ex["decode_score_new"]):
                if score > 0.9:
                    data[key_input][key_output]["input_fake_ident"].append(ex["input_fake_ident"][i])
                    data[key_input][key_output]["decode_conclusion"].append(ex["decode_conclusion"][i])
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

class StepwiseDataset(Dataset):  # type: ignore
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
        hard_type: str,
        hard_prob: float,
        path_train_contrast: str,
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
        self.hard_type = hard_type
        self.hard_prob = hard_prob
        # debug here
        if is_train and hard_type == "enhanced":
            self.cont_ex = read_entailmentbank_contrast_proofs(path_train_contrast, is_train=True)

        self.data = read_entailmentbank_proofs(path, is_train)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        ex = self.data[idx]
        if self.is_train:
            return self.get_example_train(ex)
        else:
            return self.get_example_eval(ex)

    def collate(self, examples: List[Example]) -> Batch:
        inp = [ex["input_seq"] for ex in examples]
        input_seq = self.tokenizer(
            inp,
            padding="longest",
            max_length=self.max_input_len,
            truncation=True,
            return_tensors="pt",
        )

        batch = {
            "input_seq": inp,
            "input_seq_ids": input_seq.input_ids,
            "input_seq_mask": input_seq.attention_mask,
        }
        for k in examples[0].keys():
            if k not in ("input_seq", "output_seq"):
                batch[k] = [ex[k] for ex in examples]

        if self.is_train:
            oup = [ex["output_seq"] for ex in examples]
            output_seq = self.tokenizer(
                oup,
                padding="longest",
                max_length=self.max_output_len,
                truncation=True,
                return_tensors="pt",
            )
            pad_tokens = torch.zeros((len(oup), 1), dtype=torch.int8)
            lm_labels = deepcopy(output_seq.input_ids)
            lm_labels = torch.cat((lm_labels, pad_tokens), 1)
            lm_labels[lm_labels == self.tokenizer.pad_token_id] = -100

            # decoder input ids
            bos_tokens = torch.zeros((len(oup), 1), dtype=torch.int8)
            decoder_input_ids = output_seq.input_ids
            decoder_input_ids = torch.cat((bos_tokens, decoder_input_ids), 1)

            decoder_attention_mask = output_seq.attention_mask
            att_tokens = torch.ones((len(oup), 1), dtype=torch.int8)
            decoder_attention_mask = torch.cat((att_tokens, decoder_attention_mask), 1)

            oup_fake_seq = [ex["output_fake_seq"] for ex in examples]
            output_fake_seq = self.tokenizer(
                oup_fake_seq,
                padding="longest",
                max_length=self.max_output_len,
                truncation=True,
                return_tensors="pt",
            )
            decoder_fake_input_ids = output_fake_seq.input_ids
            decoder_fake_input_ids = torch.cat((bos_tokens, decoder_fake_input_ids), 1)
            decoder_fake_attention_mask = output_fake_seq.attention_mask
            decoder_fake_attention_mask = torch.cat((att_tokens, decoder_fake_attention_mask), 1)

            batch["output_seq"] = oup
            batch["output_seq_ids"] = decoder_input_ids
            batch["output_seq_mask"] = decoder_attention_mask
            batch["lm_labels"] = lm_labels

            batch["output_fake_seq_ids"] = decoder_fake_input_ids
            batch["output_fake_seq_mask"] = decoder_fake_attention_mask
        return batch

    def get_example_train(self, ex: Example) -> Example:
        proof, permutation, inv_permutation = ex["proof"].shuffle_context()

        # Sample the proof step
        tree = proof.to_tree()
        int_node = random.choice(get_internal_nodes(tree))

        # Sample the goal
        if self.sample_goal == "hypothesis":
            goal_node = tree.get_tree_root()
        else:
            assert self.sample_goal == "intermediates"
            ancestors = int_node.get_ancestors()
            assert int_node not in ancestors
            ancestors.append(int_node)
            goal_node = random.choice(ancestors)

        # Sample the partial proof
        proved_subtrees = [node for node in int_node.children if not node.is_leaf()]
        if int_node is not goal_node:
            unproved_child = int_node
            for node in int_node.iter_ancestors():
                for child in node.children:
                    if child is unproved_child or child.is_leaf():
                        continue
                    if self.subtree_proved_all_or_none:
                        if random.random() < self.subtree_proved_prob:
                            proved_subtrees.append(child)
                    else:
                        proved_subtrees.extend(
                                collect_proved_subtrees(child, self.subtree_proved_prob))
                if node is goal_node:
                    break
                else:
                    unproved_child = node
        proved_subtrees.reverse()
        random.shuffle(proved_subtrees)
        partial_proof = " ".join(serialize(t) for t in proved_subtrees)

        # goal context
        input_seq = f"$hypothesis$ = {goal_node.sent} ; $context$ = {proof.serialize_context()} ; $proof$ = {partial_proof}"

        premises = [node.name for node in int_node.children]
        inv_premises = [proof.back_proof(premise, permutation) for premise in premises]
        output_key = " & ".join(sorted(inv_premises))

        random.shuffle(premises)

        output_seq = " & ".join(premises)
        if goal_node is int_node:
            output_seq = output_seq + " -> hypothesis;"
        else:
            output_seq = output_seq + f" -> int: {int_node.sent};"

        # find contrast example here
        if self.hard_type == "vanilla":
            neg_fake_id = random.choice(premises)
            neg_fake_seq = " & ".join(premises) + " -> int: " + proof.ident2sent(neg_fake_id)
        elif self.hard_type == "enhanced":
            input_key = f"$context$ = {ex['proof'].serialize_context()}"
            if len(self.cont_ex[input_key][output_key]["input_fake_ident"]) == 0 or random.random() < self.hard_prob:
                neg_fake_id = random.choice(premises)
                neg_fake_seq = " & ".join(premises) + " -> int: " + proof.ident2sent(neg_fake_id)
            else:
                neg_fake_id = np.random.randint(0, len(self.cont_ex[input_key][output_key]["input_fake_ident"]))
                fake_premises = deepcopy(self.cont_ex[input_key][output_key]["input_fake_ident"][neg_fake_id])
                fake_conclusion = self.cont_ex[input_key][output_key]["decode_conclusion"][neg_fake_id]
                random.shuffle(fake_premises)
                if set(fake_premises) == set(premises):
                    neg_fake_id = random.choice(premises)
                    neg_fake_seq = " & ".join(premises) + " -> int: " + ex["proof"].ident2sent(neg_fake_id)
                else:
                    neg_fake_seq = " & ".join(fake_premises) + " -> int: " + fake_conclusion
                    neg_fake_seq = proof.shuffle_proof(neg_fake_seq, inv_permutation) 

        train_ex = {}
        train_ex["input_seq"] = input_seq
        train_ex["output_seq"] = output_seq
        train_ex["output_fake_seq"] = neg_fake_seq
        return train_ex

    def get_example_eval(self, ex: Example) -> Example:
        proof = ex["proof"]
        context_text = proof.serialize_context()
        input_seq = f"$hypothesis$ = {proof.hypothesis} ; $context$ = {context_text} ; $proof$ = "

        ex = deepcopy(ex)
        ex["input_seq"] = input_seq
        return ex


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
        hard_type: str,
        hard_prob: float,
        path_train: str,
        path_train_contrast: str,
        path_val: str,
        path_test: str,
        subtree_proved_prob: float,
        subtree_proved_all_or_none: bool,
    ) -> None:
        super().__init__()
        assert dataset in ("entailmentbank")
        assert hard_type in ("vanilla", "enhanced")
        self.dataset = dataset
        self.stepwise = stepwise
        self.sample_goal = sample_goal
        self.model_name = model_name
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.hard_type = hard_type
        self.hard_prob = hard_prob
        self.path_train = path_train
        self.path_train_contrast = path_train_contrast
        self.path_val = path_val
        self.path_test = path_test
        self.subtree_proved_prob = subtree_proved_prob
        self.subtree_proved_all_or_none = subtree_proved_all_or_none

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
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
                self.hard_type,
                self.hard_prob,
                self.path_train_contrast,
            )

        if stage in (None, "fit", "validate"):
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
                self.hard_type,
                self.hard_prob,
                self.path_train_contrast,
            )

        if stage in (None, "test"):
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
                self.hard_type,
                self.hard_prob,
                self.path_train_contrast,
            )

    def train_dataloader(self) -> DataLoader:  # type: ignore
        return DataLoader(
            self.ds_train,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.ds_train.collate,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore
        return DataLoader(
            self.ds_val,
            1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.ds_val.collate,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore
        return DataLoader(
            self.ds_test,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.ds_test.collate,
            pin_memory=True,
            drop_last=False,
        )


