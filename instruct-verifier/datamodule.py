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

class EntailmentDataset(Dataset):
    def __init__(
        self,
        path: str,
        model_name: str,
        split: str,
        max_input_len: int
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length=max_input_len
        )
        # for gpt2
        self.tokenizer.pad_token = self.tokenizer.eos_token
        assert split in ("train", "val")
        self.split = split
        self.max_input_len = max_input_len
        self.data = self.preprocess(path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        ex = self.data[idx]
        premises = deepcopy(ex["premises"])
        conclusion = deepcopy(ex["conclusion"])
        return {
            "premises": f"$premises$: {premises}",
            "conclusion": f"$conclusion$: {conclusion}",
        }

    def preprocess(self, path:str) -> List[Example]:
        #raise NotImplementedError
        data = []
        for count, line in enumerate(open(path)):
            info = json.loads(line.strip())
            if info['label']:
                data.append(info)
            # filter by label here
            if count == 20:
                break
            #data.append(info)
        return data

    def collate(self, examples: List[Example]) -> Batch:
        inp = [ex["premises"] + " " + ex["conclusion"] + self.tokenizer.eos_token for ex in examples]
        
        entailment = self.tokenizer(
            inp,
            padding="longest",
            truncation=True,
            max_length=self.max_input_len,
            return_tensors="pt",
        )

        input_ids = entailment.input_ids
        labels = input_ids.clone()
        input_exp = [ex["premises"] + " $conclusion$: " for ex in examples]
        input_exp_seq = self.tokenizer(
            input_exp,
            padding="longest",
            truncation=True,
            max_length=self.max_input_len,
            return_tensors="pt",
        )
        
        input_attention_mask = entailment.attention_mask
        for i in range(input_exp_seq.attention_mask.shape[0]):
            labels[i, :sum(input_exp_seq.attention_mask[i])] = -100 #ignore index
        return {
            "premises": [ex["premises"] for ex in examples],
            "conclusion": [ex["conclusion"] for ex in examples],
            "input_ids": entailment["input_ids"],
            "attention_mask": entailment["attention_mask"],
            "label": labels,
            "input_seq": input_exp,
        }


class EntailmentDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        path_train: str,
        path_val: str,
        model_name: str,
        batch_size: int,
        num_workers: int,
        max_input_len: int,
        max_output_len: int
    ) -> None:
        super().__init__()
        self.dataset = EntailmentDataset
        self.path_train = path_train
        self.path_val = path_val
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.ds_train = self.dataset(
                self.path_train,
                self.model_name,
                "train",
                self.max_input_len
            )

        if stage in (None, "fit", "validate"):
            self.ds_val = self.dataset(
                self.path_val,
                self.model_name,
                "val",
                self.max_input_len
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
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.ds_val.collate,
            pin_memory=True,
            drop_last=False
        )


if __name__ == "__main__":

    path_train = "/home/ysuay/codes/LLM+Reason/codes/LLMProofs/data_prompt/entailment_verifier_task2_samples.txt"
    path_val = "/home/ysuay/codes/LLM+Reason/codes/LLMProofs/data_prompt/entailment_verifier_task2_samples.txt"

    ds_val = EntailmentDataset(
        split="val",
        path=path_train,
        model_name="gpt2-large",
        max_input_len=1024)
    print(len(ds_val))
    print(ds_val[0])
    ret = ds_val.collate([ds_val[0], ds_val[1]])
    print(ret)
