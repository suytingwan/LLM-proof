"""
Dataloading for EntailmentBank and RuleTaker.
"""
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
    data = defaultdict(list)
    num_invalid = 0

    for count, line in enumerate(open(path)):
        ex = json.loads(line)
        hypothesis = ex["hypothesis"]
        context = ex["context"]
        proof_text = ex["proof_gt"].strip()
        #if count == 10:
        #    break
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
            info["input_fake_ident"] = ex["input_fake_ident"]
            info["partial_proof"] = ex["partial_proof"]
            info["input_seq"] = ex["input_seq"]
            info["stepwise_goal"] = ex["stepwise_goal"]
            info["hypothesis_sample"] = ex["hypothesis_sample"]
            info["decode_conclusion"] = ex["decode_conclusion"]
            info["decode_score"] = ex["decode_score"]
            data[ex["proof_id"]].append(info)
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
            info["hypothesis_sample"] = ex["hypothesis_sample"]
            info["out_mask_scores"] = ex["out_mask_scores"]
            info["decode_conclusion"] = ex["decode_conslusion"]
            info["answer"] = ex["answer"]
            info["depth"] = ex["depth"]
            data.append(info)
        except InvalidProofStep:
            assert is_train
            num_invalid += 1

    print(f"{len(data)} proofs loaded. {num_invalid} invalid ones removed.")
    return data

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
                self.proof_ids = list(self.data.keys())
            else:
                self.data = read_entailmentbank_proofs(path, is_train)
        else:
            assert dataset == "ruletaker"
            if self.is_train:
                self.data = read_ruletaker_proofs(path, is_train)
            else:
                self.data = read_ruletaker_proofs(path, is_train)

    def __len__(self) -> int:
        if self.is_train:
            return len(self.proof_ids)
        else:
            return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        if self.is_train:
            index = random.randint(0, len(self.proof_ids)-1)
            key = self.proof_ids[index]
            index2 = random.randint(0, len(self.data[key])-1)
            ex = self.data[key][index2]
            return self.get_example_train(ex)
        else:
            ex = self.data[idx]
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
            inp_fake = [ex["input_fake_seq"] for ex in examples]
            input_fake_seq = self.tokenizer(
                inp_fake,
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

            oup_standard_pos = [ex["positive_standard_seq"] for ex in examples]
            output_standard_pos = self.tokenizer(
                oup_standard_pos,
                padding="longest",
                max_length=self.max_output_len,
                truncation=True,
                return_tensors="pt",
            )
            output_standard_pos.input_ids[
                output_standard_pos.input_ids == self.tokenizer.pad_token_id
            ] = -100

            batch["input_fake_seq"] = inp_fake
            batch["input_fake_seq_ids"] = input_fake_seq.input_ids
            batch["input_fake_seq_mask"] = input_fake_seq.attention_mask

            batch["output_seq"] = oup
            batch["output_seq_ids"] = output_seq.input_ids
            batch["output_seq_mask"] = output_seq.attention_mask

            batch["output_seq_pos"] = oup_pos
            batch["output_seq_pos_ids"] = output_seq_pos.input_ids
            batch["output_seq_pos_mask"] = output_seq_pos.attention_mask

            batch["output_standard_pos"] = oup_standard_pos
            batch["output_standard_pos_ids"] = output_standard_pos.input_ids
            batch["output_standard_pos_mask"] = output_standard_pos.attention_mask
        return batch

    def get_example_train(self, ex: Example) -> Example:
        proof = ex["proof"]
        partial_proof = ex["partial_proof"]
        hypothesis_sample = ex["hypothesis_sample"]
        # already with paritial proof
        input_seq = ex["input_seq"]

        # reference output_seq, positive output_seq, negative output_seq
        output_seq = ex["stepwise_goal"]
        # win rate? rerank the answer by verifier

        tmp_score = []
        for i in range(len(ex["decode_score"][0])):
            if (i+1) % 3 == 0:
                pass
            else:
                tmp_score.append(ex["decode_score"][0][i])
        if max(tmp_score) <= 0.7:
            pos_index = np.argmax(ex["decode_score"][0])
        else:
            while True:
                pos_index = random.randint(0, len(ex["decode_score"][0])-1)
                #if ex['decode_score'][0][pos_index] >= 0.6:
                #if ex['decode_score'][0][pos_index] >= 0.7:
                #if ex['decode_score'][0][pos_index] >= 0.8:
                if (pos_index+1) % 3 != 0 and ex['decode_score'][0][pos_index] >= 0.7:
                    break

        positive_fake_seq = ex["input_fake_ident"][pos_index]
        #pos_standard = positive_fake_seq + " -> int: " + ex["decode_conclusion"][pos_index] + ";"
        if ex['decode_score'][0][pos_index] >= 0.9:
            pos_standard = positive_fake_seq + " -> hypothesis;"
        else:
            pos_standard = positive_fake_seq + " -> int: " + ex["decode_conclusion"][pos_index] + ";"


        if random.random() < 0.1:
            new_sub_goal = ex["decode_conclusion"][pos_index]
            input_fake_seq = f"$hypothesis$ = {new_sub_goal} ; $context$ = {proof.serialize_context()} ; $proof$ = {partial_proof}"
            positive_output_seq = positive_fake_seq + " -> hypothesis;"
        else:
            input_fake_seq = input_seq
            positive_output_seq = pos_standard

        train_ex = {}
        train_ex["proof"] = proof
        train_ex["input_seq"] = input_seq
        train_ex["output_seq"] = output_seq
        train_ex["input_fake_seq"] = input_fake_seq
        train_ex["positive_output_seq"] = positive_output_seq
        train_ex["positive_standard_seq"] = pos_standard
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


if __name__ == "__main__":
  
    path_val = "/home/ysuay/codes/LLM+Reason/codes/LLMProofs/sample_sft/eval_entailmentbank_task2/lightning_logs/version_8/results_train.json"

    ds_train = StepwiseDataset(
        dataset="entailmentbank",
        path=path_val,
        model_name="t5-large",
        max_input_len=1024,
        max_output_len=64,
        is_train=True)

    #ds_train = StepwiseDataset(
    #    dataset="entailmentbank",
    #    path=path_val,
    #    model_name="t5-large",
    #    max_input_len=1024,
    #    max_output_len=64,
    #    is_train=False)

    print(len(ds_train))
    #for i in range(len(ds_train)):
    #    print(ds_train[i]["output_seq"], ds_train[i]["positive_output_seq"], ds_train[i]["negative_output_seq"], ds_train[i]["top1_out_seq"])
    print(ds_train[0])
    print(ds_train[1])
    #print(ds_train.collate([ds_train[0], ds_train[1]]))

