from common import *
from verifier.model import EntailmentClassifier
from prover.proof import ProofStep, Proof, InvalidProofStep
from prover.search import ProofGraph
import numpy as np
import os
import json
import torch
import itertools
import pytorch_lightning as pl
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    AutoModelForCausalLM,
    BartForConditionalGeneration,
    LogitsProcessor,
)
from prover.evaluate import evaluate_entailmentbank, evaluate_ruletaker


# Some handcrafted heuristics for constraining the predicted proof steps.
# They often make the proof graph less cluttered but do not improve the final performance.
# So we do not use them by default.
class PermutationInvarianceLogitsProcessor(LogitsProcessor):
    def __init__(
        self, num_beams: int, context: List[OrderedDict[str, str]], tokenizer: Any
    ) -> None:
        self.num_beams = num_beams
        self.context = context
        self.tokenizer = tokenizer
        self.semicolon_token_id = tokenizer.convert_tokens_to_ids(";")

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        generated_texts = self.tokenizer.batch_decode(
            input_ids, skip_special_tokens=True
        )

        batch_size = input_ids.size(0) // self.num_beams
        unique_premises: List[Set[Any]] = [set() for _ in range(batch_size)]

        for i, prefix in enumerate(generated_texts):
            if "->" in prefix:  # conclusion
                if prefix.count("->") > 1:
                    scores[i, :] = float("-inf")
                    continue
                concl = prefix.split("->")[1].strip()
                if concl == "hypothesis":
                    # Only ";" after "-> hypothesis".
                    s = scores[i, self.semicolon_token_id].item()
                    scores[i, :] = float("-inf")
                    scores[i, self.semicolon_token_id] = s
                elif ";" in concl:
                    # Must end after ";"
                    s = scores[i, self.tokenizer.eos_token_id].item()
                    scores[i, :] = float("-inf")
                    scores[i, self.tokenizer.eos_token_id] = s
                elif (
                    concl != ""
                    and not concl.startswith("int")
                    and not "int".startswith(concl)
                ):
                    # The conclusion is either the hypothesis or an intermediate.
                    scores[i, :] = float("-inf")
                elif "-> int" in prefix:
                    # Only one conclusion for fixed premises.
                    j = scores[i, :].argmax()
                    s = scores[i, j].item()
                    scores[i, :] = float("-inf")
                    scores[i, j] = s

            else:  # premises
                n = i // self.num_beams
                premises = tuple(sorted([p.strip() for p in prefix.split("&")]))
                if premises in unique_premises[n] or len(set(premises)) < len(premises):
                    scores[i, :] = float("-inf")
                    continue
                unique_premises[n].add(premises)

                tokens = prefix.split()
                for t in tokens[:-1]:
                    if t != "&" and re.fullmatch(r"(int|sent)\d+", t) == None:
                        scores[i, :] = float("-inf")
                    elif (
                        re.fullmatch(r"sent\d+", t) != None and t not in self.context[n]
                    ):
                        scores[i, :] = float("-inf")
                if len(tokens) >= 1:
                    t = tokens[-1]
                    if (
                        t != "&"
                        and re.fullmatch(r"(int|sent)\d+", t) == None
                        and not "sent".startswith(t)
                        and not "int".startswith(t)
                    ):
                        scores[i, :] = float("-inf")

        return scores


class EntailmentWriter(pl.LightningModule):
    def __init__(
        self,
        dataset: str,
        stepwise: bool,
        max_num_steps: int,
        model_name: str,
        lr: float,
        warmup_steps: int,
        num_beams: int,
        topk: int,
        max_input_len: int,
        proof_search: bool,
        verifier_weight: float,
        verifier_ckpt: Optional[str] = None,
        oracle_prover: Optional[bool] = False,
        oracle_verifier: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.stepwise = stepwise
        self.max_num_steps = max_num_steps
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_beams = num_beams
        self.topk = topk
        self.verifier_weight = verifier_weight
        self.proof_search = proof_search
        self.oracle_prover = oracle_prover
        self.oracle_verifier = oracle_verifier
        if stepwise and verifier_weight > 0:
            assert verifier_weight <= 1.0
            assert verifier_ckpt is not None
            self.verifiers = [
                EntailmentClassifier.load_from_checkpoint(verifier_ckpt)
            ]  # Avoid making the verifier a submodule.

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length=max_input_len
        )
        if (
            model_name.startswith("t5-")
            or model_name.startswith("google/t5-v1_1-")
            or model_name.startswith("google/byt5-")
            or model_name.startswith("google/flan-t5")
            or model_name.startswith("gpt")
        ):
            self.seq2seq = T5ForConditionalGeneration.from_pretrained(model_name)
            #self.seq2seq = T5ForConditionalGeneration.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype="bfloat16")
            #self.seq2seq = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        elif model_name.startswith("facebook/bart-"):
            self.seq2seq = BartForConditionalGeneration.from_pretrained(model_name)
        else:
            raise NotImplementedError

    def forward(  # type: ignore
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Any:
        return self.seq2seq(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss

    def move_verifier_to_device(self) -> None:
        if hasattr(self, "verifiers"):
            self.verifiers[0].to(self.device)

    def on_train_start(self) -> None:
        self.move_verifier_to_device()
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)  # type: ignore
            assert self.trainer is not None
            print(f"Logging to {self.trainer.log_dir}")

    def on_validation_start(self) -> None:
        self.move_verifier_to_device()

    def on_test_start(self) -> None:
        self.move_verifier_to_device()

    def generate_entire_proof(
        self, input_text: List[str]
    ) -> Tuple[List[str], List[float]]:
        """
        Single-shot proof generation with text-to-text transformers.
        """
        assert self.trainer is not None
        input = self.tokenizer(
            input_text,
            padding="longest",
            max_length=self.trainer.datamodule.max_input_len,  # type: ignore
            truncation=True,
            return_tensors="pt",
        )
        output = self.seq2seq.generate(
            input_ids=input.input_ids.to(self.device, non_blocking=True),
            attention_mask=input.attention_mask.to(self.device, non_blocking=True),
            max_length=self.trainer.datamodule.max_output_len,  # type: ignore
            num_beams=self.num_beams,
            num_return_sequence=1,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

        output_text = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )
        scores = output.sequences_scores.detach().exp().tolist()
        return output_text, scores

    def generate_proof_step(
        self,
        input_text: List[str],
    ) -> Tuple[List[str], Any]:
        """
        Generate a single proof step with text-to-text transformers.
        """
        input = self.tokenizer(
            input_text,
            padding="longest",
            max_length=self.trainer.datamodule.max_input_len,
            truncation=True,
            return_tensors="pt",
        )

        output = self.seq2seq.generate(
            input_ids=input.input_ids.to(self.device, non_blocking=True),
            attention_mask=input.attention_mask.to(self.device, non_blocking=True),
            max_length=self.trainer.datamodule.max_output_len,
            num_beams=self.num_beams,
            num_return_sequences=self.topk,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True
        )
        output_text = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )
        output_text = [output_text_.strip().split("$proof$ = ")[-1] for output_text_ in output_text]

        batch_size = len(input_text)
        assert len(output_text) % batch_size == 0
        k = len(output_text) // batch_size
        output_text = [output_text[i*k : (i+1)*k] for i in range(batch_size)]

        output_scores = output.sequences_scores.detach().exp().cpu().numpy()
        assert 0.0 <= output_scores.min() <= output_scores.max() <= 1.0
        output_scores = [output_scores[i*k : (i+1)*k] for i in range(batch_size)]
        return output_text, output_scores


    def normalize_predicted_step(self, step: str, proof: Proof) -> str:
        if "-> int:" in step:
            step = step.replace("-> int:", f"-> {proof.next_int()}:").strip()
        return step

    def filter_invalid_steps(
        self,
        output_text: List[str],
        output_scores: List[float],
        proofs: List[Proof],
        strict: bool,
    ) -> Tuple[List[List[ProofStep]], List[List[float]]]:
        batch_size = len(proofs)

        all_proof_steps = [[] for _ in range(batch_size)]
        all_scores = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            assert len(output_text[i]) == len(output_scores[i])

            for text, score in zip(output_text[i], output_scores[i]):
                idx = text.find(";")
                if idx != -1:
                    text = text[:idx]
                else:
                    continue
                s = self.normalize_predicted_step(text, proofs[i])
                try:
                    step = ProofStep(proofs[i], s, strict)
                except InvalidProofStep:
                    continue
                all_proof_steps[i].append(step)
                all_scores[i].append(float(score))

        return all_proof_steps, all_scores

    def training_step(self, batch: Batch, batch_idx: int) -> Dict[str, torch.Tensor]:  # type: ignore
        if self.stepwise:
            loss = self(
                batch["input_seq_ids"], batch["input_seq_mask"], batch["output_seq_ids"]
            )
            #self.log("loss_train", loss, on_epoch=True, sync_dist=True)
            self.log("loss_train", loss, on_epoch=True, rank_zero_only=True)
        else:
            loss = self(
                batch["input_seq_ids"],
                batch["input_seq_mask"],
                batch["output_seq_ids"],
            )
            #self.log("loss_train", loss, on_epoch=True, sync_dist=True)
            self.log("loss_train", loss, on_epoch=True, rank_zero_only=True)

        return {"loss": loss}

    def validation_step(self, batch: Batch, batch_idx: int) -> Tuple[Any]:  # type: ignore
        return self.val_test_step("val", batch, batch_idx)

    def test_step(self, batch: Batch, batch_idx: int) -> Tuple[Any]:  # type: ignore
        return self.val_test_step("test", batch, batch_idx)

    def validation_epoch_end(self, outputs: Iterable[Any]) -> None:
        print("validate end ...")
        pass

    def test_epoch_end(self, outputs: Iterable[Any]) -> None:
        pass

    def val_test_step(self, split: str, batch: Batch, batch_idx: int) -> None:
        #generate top-k proof
        output_text, output_scores = self.generate_proof_step(batch["input_seq"])
        #print(output_text)
        #print(output_scores)

        json_path = os.path.join(self.trainer.log_dir, f"results_{split}.json")
        fw = open(json_path, 'a+')
        if self.dataset == "entailmentbank":
            for out_text, out_score, proof, partial_proof, stepwise_goal in \
                zip(output_text, output_scores, batch["proof"], batch["partial_proof"], batch["output_seq"]):
                ret = {
                        "proof_candidates": out_text,
                        "score_candidates": out_score.tolist(),
                        "proof_id": proof.proof_id,
                        "hypothesis": proof.hypothesis,
                        "context": proof.context,
                        "proof_gt": proof.proof_text,
                        "partial_proof": partial_proof,
                        "stepwise_goal": stepwise_goal,
                      }
                fw.write(json.dumps(ret) + '\n')
        else:
            for out_text, out_score, proof, answer, depth, all_proof, partial_proof, stepwise_goal in \
                zip(output_text, output_scores, batch["proof"], batch["answer"], batch["depth"], batch["all_proofs"], batch["partial_proof"], batch["output_seq"]):
                ret = {
                        "proof_candidates": out_text,
                        "score_candidates": out_score.tolist(),
                        "proof_id": proof.proof_id,
                        "hypothesis": proof.hypothesis,
                        "context": proof.context,
                        "proof_gt": proof.proof_text,
                        "answer": answer,
                        "depth": depth,
                        "all_proof": all_proof,
                        "partial_proof": partial_proof,
                        "stepwise_goal": stepwise_goal,
                      }
                fw.write(json.dumps(ret) + '\n')
        fw.close()

    def configure_optimizers(self) -> Dict[str, Any]:
        assert self.trainer is not None
        if self.trainer.max_steps != -1:
            max_steps = self.trainer.max_steps
        else:
            max_steps = (
                self.trainer.max_epochs
                * len(self.trainer.datamodule.train_dataloader())  # type: ignore
                // self.trainer.accumulate_grad_batches
            )
        return get_optimizers(
            self.parameters(),
            self.lr,
            self.warmup_steps,
            max_steps,
        )
