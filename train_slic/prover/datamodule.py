"""
Dataloading for EntailmentBank and RuleTaker.
"""
import sys
sys.path.append("../")

from copy import deepcopy
from common import *
from prover.proof import Proof, InvalidProofStep
import random
import json
import numpy as np
import itertools
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer


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
            #if not is_train and count == 5:
            #    break
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

def read_entailmentbank_slic_proofs(path: str, is_train: bool) -> List[Example]:
    """
    Load the EntailmentBank dataset.
    """
    data = []
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
            info = {}
            info["proof"] = proof
            info["verifier_loss"] = ex["verifier_loss"]
            info["proof_candidates"] = ex["proof_candidates"]
            info["score_candidates"] = ex["score_candidates"]
            info["partial_proof"] = ex["partial_proof"]
            info["stepwise_goal"] = ex["stepwise_goal"]
            info["out_mask_scores"] = ex["out_mask_scores"]
            data.append(info)
        except InvalidProofStep:
            assert is_train
            num_invalid += 1

    print(f"{len(data)} proofs loaded. {num_invalid} invalid ones removed.")
    return data

def read_ruletaker_proofs(path: str, is_train: bool) -> List[Example]:
    """
    Load the RuleTaker dataset.
    """
    data = []
    num_invalid = 0

    for line in open(path):
        ex = json.loads(line)
        hypothesis = normalize(ex["hypothesis"])
        context = extract_context(ex["context"])

        if is_train:
            for proof in ex["proofs"]:
                try:
                    prf = Proof(
                        context,
                        hypothesis,
                        normalize(proof.strip()),
                        strict=True,
                        requires_complete=True,
                    )
                    ans = ex["answer"]
                    assert ans == True
                    data.append(
                        {
                            "answer": ans,
                            "depth": ex["depth"],
                            "proof": prf,
                            "all_proofs": ex["proofs"],
                        }
                    )
                except InvalidProofStep:
                    num_invalid += 1

        else:
            proof = ex["proofs"][0] if len(ex["proofs"]) > 0 else ""
            data.append(
                {
                    "answer": ex["answer"],
                    "depth": ex["depth"],
                    "proof": Proof(
                        context,
                        hypothesis,
                        proof,
                        strict=False,
                        requires_complete=False,
                    ),
                    "all_proofs": ex["proofs"],
                }
            )
            if ex["answer"] == "Unknown":
                ans = "Unknown"
            else:
                ans = not ex["answer"]
            data.append(
                {
                    "answer": ans,
                    "depth": ex["depth"],
                    "proof": Proof(
                        context,
                        f"i don't think {hypothesis}",
                        proof,
                        strict=False,
                        requires_complete=False,
                    ),
                    "all_proofs": ex["proofs"],
                }
            )

    print(f"{len(data)} proofs loaded. {num_invalid} invalid ones removed.")
    return data


def read_ruletaker_slic_proofs(path: str, is_train: bool) -> List[Example]:
    """
    Load the EntailmentBank dataset.
    """
    data = []
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
            info = {}
            info["proof"] = proof
            info["verifier_loss"] = ex["verifier_loss"]
            info["proof_candidates"] = ex["proof_candidates"]
            info["score_candidates"] = ex["score_candidates"]
            info["partial_proof"] = ex["partial_proof"]
            info["stepwise_goal"] = ex["stepwise_goal"]
            info["out_mask_scores"] = ex["out_mask_scores"]
            info["answer"] = ex["answer"]
            info["depth"] = ex["depth"]
            data.append(info)
        except InvalidProofStep:
            assert is_train
            num_invalid += 1

    print(f"{len(data)} proofs loaded. {num_invalid} invalid ones removed.")
    return data


class EntireProofsDataset(Dataset):  # type: ignore
    def __init__(
        self,
        dataset: str,
        path: str,
        model_name: str,
        max_input_len: int,
        max_output_len: int,
        is_train: bool,
    ) -> None:
        super().__init__()
        max_len = max(max_input_len, max_output_len)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length=max_len
        )
        #self.tokenizer = AutoTokenizer.from_pretrained(
        #    '../tulu-7b', model_max_length=max_input_len
        #)
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.is_train = is_train
        if dataset == "entailmentbank":
            if self.is_train:
                self.data = read_entailmentbank_slic_proofs(path, is_train)
            else:
                self.data = read_entailmentbank_proofs(path, is_train)
        else:
            assert dataset == "ruletaker"
            if self.is_train:
                self.data = read_ruletaker_slic_proofs(path, is_train)
            else:
                self.data = read_ruletaker_proofs(path, is_train)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        ex = self.data[idx]
        proof = ex["proof"]
        if self.is_train:
            proof = proof.shuffle_context()

        input_seq = f"$hypothesis$ = {proof.hypothesis} ; $context$ = {proof.serialize_context()}"

        ex = deepcopy(ex)
        ex["input_seq"] = input_seq
        ex["output_seq"] = proof.proof_text
        return ex

    def collate(self, examples: List[Example]) -> Batch:
        inp = [ex["input_seq"] for ex in examples]
        input_seq = self.tokenizer(
            inp,
            padding="longest",
            max_length=self.max_input_len,
            truncation=True,
            return_tensors="pt",
        )

        oup = [ex["output_seq"] for ex in examples]
        output_seq = self.tokenizer(
            oup,
            padding="longest",
            max_length=self.max_output_len,
            truncation=True,
            return_tensors="pt",
        )
        output_seq.input_ids[output_seq.input_ids == self.tokenizer.pad_token_id] = -100

        batch = {
            "input_seq": inp,
            "input_seq_ids": input_seq.input_ids,
            "input_seq_mask": input_seq.attention_mask,
            "output_seq": oup,
            "output_seq_ids": output_seq.input_ids,
            "output_seq_mask": output_seq.attention_mask,
        }
        for k in examples[0].keys():
            if k not in ("input_seq", "output_seq"):
                batch[k] = [ex[k] for ex in examples]
        return batch


class StepwiseDataset(Dataset):  # type: ignore
    def __init__(
        self,
        dataset: str,
        path: str,
        model_name: str,
        max_input_len: int,
        max_output_len: int,
        is_train: bool,
    ) -> None:
        super().__init__()
        max_len = max(max_input_len, max_output_len)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length=max_len
        )
        #self.tokenizer = AutoTokenizer.from_pretrained(
        #    '../tulu-7b', model_max_length=max_input_len
        #)
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.is_train = is_train
        if dataset == "entailmentbank":
            if self.is_train:
                self.data = read_entailmentbank_slic_proofs(path, is_train)
            else:
                self.data = read_entailmentbank_proofs(path, is_train)
        else:
            assert dataset == "ruletaker"
            if self.is_train:
                self.data = read_ruletaker_proofs(path, is_train)
            else:
                self.data = read_ruletaker_proofs(path, is_train)

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
            output_seq.input_ids[
                output_seq.input_ids == self.tokenizer.pad_token_id
            ] = -100

            oup_pos = [ex["positive_output_seq"] for ex in examples]
            output_seq_pos = self.tokenizer(
                oup_pos,
                padding="longest",
                max_length=self.max_output_len,
                truncation=True,
                return_tensors="pt",
            )
            output_seq_pos.input_ids[
                output_seq_pos.input_ids == self.tokenizer.pad_token_id
            ] = -100

            oup_neg = [ex["negative_output_seq"] for ex in examples]
            output_seq_neg = self.tokenizer(
                oup_neg,
                padding="longest",
                max_length=self.max_output_len,
                truncation=True,
                return_tensors="pt",
            )
            output_seq_neg.input_ids[
                output_seq_neg.input_ids == self.tokenizer.pad_token_id
            ] = -100

            batch["output_seq"] = oup
            batch["output_seq_ids"] = output_seq.input_ids
            batch["output_seq_mask"] = output_seq.attention_mask

            batch["output_seq_pos"] = oup_pos
            batch["output_seq_pos_ids"] = output_seq_pos.input_ids
            batch["output_seq_pos_mask"] = output_seq_pos.attention_mask

            batch["output_seq_neg"] = oup_neg
            batch["output_seq_neg_ids"] = output_seq_neg.input_ids
            batch["output_seq_neg_mask"] = output_seq_neg.attention_mask
        return batch

    def get_example_train(self, ex: Example) -> Example:
        proof = ex["proof"]
        partial_proof = ex["partial_proof"]
        stepwise_goal = ex["stepwise_goal"]
        # already with paritial proof
        input_seq = f"$hypothesis$ = {stepwise_goal} ; $context$ = {proof.serialize_context()} ; $proof$ = {partial_proof}"

        # reference output_seq, positive output_seq, negative output_seq
        output_seq = ex["stepwise_goal"]
        # win rate?
        pos_index = np.argmin(ex["verifier_loss"])
        positive_output_seq = ex["proof_candidates"][pos_index]
        neg_index = np.argmax(ex["verifier_loss"])
        negative_output_seq = ex["proof_candidates"][neg_index]

        train_ex = {}
        train_ex["proof"] = proof
        train_ex["input_seq"] = input_seq
        train_ex["output_seq"] = output_seq
        train_ex["positive_output_seq"] = positive_output_seq
        train_ex["negative_output_seq"] = negative_output_seq
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
        model_name: str,
        max_input_len: int,
        max_output_len: int,
        batch_size: int,
        num_workers: int,
        path_train: str,
        path_val: str,
        path_test: str,
    ) -> None:
        super().__init__()
        assert dataset in ("entailmentbank", "ruletaker")
        self.dataset = dataset
        self.stepwise = stepwise
        self.model_name = model_name
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.path_train = path_train
        self.path_val = path_val
        self.path_test = path_test

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
                    is_train=True,
                )
            else:
                self.ds_train = EntireProofsDataset(  # type: ignore
                    self.dataset,
                    self.path_train,
                    self.model_name,
                    self.max_input_len,
                    self.max_output_len,
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
                    is_train=False,
                )
            else:
                self.ds_val = EntireProofsDataset(  # type: ignore
                    self.dataset,
                    self.path_val,
                    self.model_name,
                    self.max_input_len,
                    self.max_output_len,
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
                    is_train=False,
                )
            else:
                self.ds_test = EntireProofsDataset(  # type: ignore
                    self.dataset,
                    self.path_test,
                    self.model_name,
                    self.max_input_len,
                    self.max_output_len,
                    is_train=False,
                )

    def train_dataloader(self) -> DataLoader:  # type: ignore
        return DataLoader(
            self.ds_train,
            self.batch_size,
            shuffle=True,
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


if __name__ == "__main__":
  
    
    #path_train = "/home/ysuay/codes/LLM+Reason/codes/LLMProofs/offline/eval_entailmentbank_task2/lightning_logs/version_4/results_train.json"
    path_val = "/home/ysuay/codes/LLM+Reason/codes/NLProofS/data/entailment_trees_emnlp2021_data_v3/dataset/task_2/dev.jsonl"

    #ds_train = StepwiseDataset(
    #    dataset="entailmentbank",
    #    path=path_train,
    #    model_name="t5-large",
    #    max_input_len=1024,
    #    max_output_len=64,
    #    is_train=True)

    ds_train = StepwiseDataset(
        dataset="entailmentbank",
        path=path_val,
        model_name="t5-large",
        max_input_len=1024,
        max_output_len=64,
        is_train=False)

    print(len(ds_train))
    print(ds_train[0])
    print(ds_train.collate([ds_train[0], ds_train[1]]))

