from common import *
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
)
from evaluate import evaluate_entailmentbank, evaluate_ruletaker
from prover.proof import ProofStep, Proof, InvalidProofStep
import torch.nn.functional as F

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
        self.delta = 2.0
        self.lambda_reg = 0.5
        self.flag = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length=max_input_len
        )

        #Tokenizer
        #self.tokenizer.pad_token = self.tokenizer.eos_token
        if (
            model_name.startswith("t5-")
            or model_name.startswith("google/t5-v1_1-")
            or model_name.startswith("google/byt5-")
            or model_name.startswith("flan-t5")
        ):
            self.seq2seq = T5ForConditionalGeneration.from_pretrained(model_name)
        elif 'gpt2' in model_name:
            self.seq2seq = AutoModelForCausalLM.from_pretrained(model)
        else:
            raise NotImplementedError

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Any:
        output = self.seq2seq(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return output.loss

    def on_train_start(self) -> None:
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)
            assert self.trainer is not None
            print(f"Logging to {self.trainer.log_dir}")

    def on_validation_start(self) -> None:
        pass

    def on_test_start(self) -> None:
        pass

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
            max_length=self.trainer.datamodule.max_input_len,
            truncation=True,
            return_tensors="pt",
        )
        output = self.seq2seq.generate(
            input_ids=input.input_ids.to(self.device, non_blocking=True),
            attention_mask=input.attention_mask.to(self.device, non_blocking=True),
            max_length=self.trainer.datamodule.max_output_len,
            num_beams=self.num_beams,
            num_return_sequences=1,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True
        )

        output_text = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )
        scores = output.sequences_scores.detach().exp().tolist()
        return output_text, scores

    def generate_stepwise_proof(
        self, proof_gt: List[Proof], batch_idx: int
    ) -> Tuple[List[str], List[float]]:
        """
        Stepwise proof generation.
        """
        proof_pred, step_scores = self.generate_greedy_proofs(proof_gt)
        proof_text_pred = [pt.proof_text for pt in proof_pred]
        score = [min(s) if len(s) > 0 else 0.0 for s in step_scores]
        return proof_text_pred, score

    def generate_proof_step(
        self, input_text: List[str], 
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

        batch_size = len(input_text)
        assert len(output_text) % batch_size == 0
        k = len(output_text) // batch_size # k predicted steps for each example
        output_text = [output_text[i*k : (i+1)*k] for i in range(batch_size)]

        output_scores = output.sequences_scores.detach().exp().cpu().numpy()
        assert 0.0 <= output_scores.min() <= output_scores.max() <= 1.0
        output_scores = [output_scores[i*k : (i+1)*k] for i in range(batch_size)]
        return output_text, output_scores

    def generate_greedy_proofs(
        self, proof_gt: List[Proof]
    ) -> Tuple[List[Proof], List[List[float]]]:
        """
        Greedily stepwise proof generation.
        """
        all_proof_pred = [
            Proof(pt.context, pt.hypothesis, proof_text="", proof_id=0, strict=True)
            for pt in proof_gt
        ]
        proof_pred = all_proof_pred
        unfinished_indexes = list(range(len(proof_gt)))
        all_step_scores: List[List[float]] = [[] for _ in proof_gt]

        for _ in range(self.max_num_steps):
            if len(unfinished_indexes) == 0:
                break
            input_text = [
                f"$hypothesis$ = {pt.hypothesis} ; $context$ = {pt.serialize_context()} ; $proof$ = {'' if pt.proof_text == '' else pt.proof_text + ';'}"
                for pt in proof_pred
            ]
            output_text, output_scores = self.generate_proof_step(input_text)

            proof_steps, prover_scores = self.filter_invalid_steps(
                output_text, output_scores, proof_pred, strict=True)

            reranked_steps, reranked_scores = proof_steps, prover_scores
            proof_steps = [
                steps[0] if len(steps) > 0 else None for steps in reranked_steps
            ]
            reranked_scores = [s[0] if len(s) > 0 else 0.0 for s in reranked_scores]

            # Execute the predicted reranked proof steps
            finished_indexes = []
            for i, j in enumerate(unfinished_indexes):
                step = proof_steps[i]
                if step is None:
                    step = self.normalize_predicted_step(
                        output_text[i][0], proof_pred[i]
                    )
                    idx = step.find(";")
                    if idx != -1:
                        step = step[:idx]
                    try:
                        step = ProofStep(proof_pred[i], step, strict=False)
                        if step.is_final():
                            finished_indexes.append(i)
                        proof_pred[i].execute(step)
                        all_step_scores[j].append(float(output_scores[i][0]))
                    except InvalidProofStep:
                        finished_indexes.append(i)
                        proof_pred[i].proof_text = "INVALID_PROOF"
                else:
                    if step.is_final():
                        finished_indexes.append(i)
                    proof_pred[i].execute(step)
                    all_step_scores[j].append(reranked_scores[i])
            unfinished_indexes = [
                j for i, j in enumerate(unfinished_indexes) if i not in finished_indexes
            ]
            proof_pred = [
                pt for i, pt in enumerate(proof_pred) if i not in finished_indexes
            ]
        
        assert (
            pt.is_complete() or pt.proof_text == "INVALID_PROOF"
            for pt in all_proof_pred
        )
        return all_proof_pred, all_step_scores
    
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

    def training_step(self, batch: Batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        if self.stepwise:
            # stepwise dataset
            loss_reference = self(
                batch["input_seq_ids"], batch["input_seq_mask"], batch["output_seq_ids"]
            )

            # wrong path due to partial proof
            #loss_fake_reference = self(
            #    batch["input_fake_seq_ids"], batch["input_fake_seq_mask"], batch["output_seq_pos_ids"]
            #)

            #unseen deductive loss
            loss_standard_pos = self(batch["input_seq_ids"], batch["input_seq_mask"], batch["output_standard_pos_ids"])

            # knowledge transfer from verifier
            #calibration_loss1 = 0.2 * loss_standard_pos
            calibration_loss1 = 0.1 * loss_standard_pos
            loss = loss_reference + calibration_loss1

            print("loss_reference: ", loss_reference)
            #print("loss_fake_reference: ", loss_fake_reference)
            print("loss_standard_pos: ", loss_standard_pos)
            loss_cal = F.relu(self.delta + loss_reference - loss_standard_pos)
            print("loss_cal: ", loss_cal)

            #if self.flag:            
            #    loss = loss_reference + loss_cal
            #    self.flag = False
            #else:
            #    loss = 0.2 * loss_standard_pos + loss_cal

            self.log("loss_train", loss, on_epoch=True, sync_dist=True)
        else:
            loss = self(
                batch["input_seq_ids"],
                batch["input_seq_mask"],
                batch["output_seq_ids"],
            )
            self.log("loss_train", loss, on_epoch=True, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, batch_idx: int) -> Tuple[Any]:
        return self.val_test_step("val", batch, batch_idx)

    def test_step(self, batch: Batch, batch_idx: int) -> Tuple[Any]:
        return self.val_test_step("test", batch, batch_idx)

    def validation_epoch_end(self, outputs: Iterable[Any]) -> None:
        return self.val_test_epoch_end("val", outputs)

    def test_epoch_end(self, outputs: Iterable[Any]) -> None:
        return self.val_test_epoch_end("test", outputs)

    def val_test_step(self, split: str, batch: Batch, batch_idx: int) -> Tuple[Any]:
        if self.stepwise:
            proof_pred, score = self.generate_stepwise_proof(batch["proof"], batch_idx)
        else:
            #print(batch["input_seq_ids"])
            #print(batch["input_seq_mask"])
            #print(batch["output_seq_ids"])
            loss = self(
                batch["input_seq_ids"],
                batch["input_seq_mask"],
                batch["output_seq_ids"],
            )
            self.log(f"loss_{split}", loss, sync_dist=True)
            proof_pred, score = self.generate_entire_proof(batch["input_seq"])

        if self.dataset == "entailmentbank":
            return proof_pred, score, batch["proof"]
        else:
            return (
                proof_pred,
                score,
                batch["proof"],
                batch["answer"],
                batch["depth"],
                batch["all_proofs"],
            )

    def val_test_epoch_end(self, split: str, outputs: Iterable[Any]) -> None:
        results = []

        for out in outputs:
            if self.dataset == "entailmentbank":
                for proof_pred, score, proof in zip(*out):
                    results.append(
                        {
                            "proof_pred": proof_pred,
                            "score": score,
                            "hypothesis": proof.hypothesis,
                            "context": proof.context,
                            "proof_gt": proof.proof_text,
                        }
                    )
            else:
                for proof_pred, score, proof, answer, depth, all_proofs in zip(*out):
                    results.append(
                        {
                            "answer": answer,
                            "depth": depth,
                            "all_proofs": all_proofs,
                            "proof_pred": proof_pred,
                            "score": score,
                            "hypothesis": proof.hypothesis,
                            "context": proof.context,
                            "proof_gt": proof.proof_text,
                        }
                    )

        assert self.trainer is not None
        if self.logger is not None and self.trainer.log_dir is not None:
            json_path = os.path.join(self.trainer.log_dir, f"results_{split}.json")
            json.dump(results, open(json_path, "wt"))
            if self.dataset == "entailmentbank":
                tsv_path = os.path.join(self.trainer.log_dir, f"results_{split}.tsv")
                with open(tsv_path, "wt") as oup:
                    for r in results:
                        #proof = r["proof_pred"].strip()
                        proof = r["proof_pred"].strip().split("$proof$ =  ")[-1]
                        if not proof.endswith(";"):
                            proof += ";"
                        oup.write(f"$proof$ = {proof}\n")
                print(f"Validation results saved to {json_path} and {tsv_path}")
            else:
                print(f"Validation results saved to {json_path}")
            # output prediction and ground truth in lines
            if self.dataset == "entailmentbank":
                tsv_path = os.path.join(self.trainer.log_dir, f"results_{split}_pred_gt.tsv")
                with open(tsv_path, "wt") as oup:
                    for r in results:
                        proof_pred = r["proof_pred"].strip()
                        if not proof_pred.endswith(";"):
                            proof_pred += ";"
                        oup.write(f"$proof_pred$ = {proof_pred}\n")
                        proof_gt = r["proof_gt"].strip()
                        if not proof_gt.endswith(";"):
                            proof_gt += ";"
                        oup.write(f"$proof_gt$ = {proof_gt}\n")
                        for key in r["context"].keys():
                            value = r["context"][key].strip()
                            oup.write(f"{key}: {value}\n")
                        hypothesis = r["hypothesis"].strip()
                        oup.write(f"$hypothesis$ = {hypothesis}\n")
                    print(f"Validation results save to {tsv_path}")
            else:
                print(f"Validation results saved to {json_path}")


        if self.dataset == "entailmentbank" and results[0]["proof_gt"] != "":
            em, f1 = evaluate_entailmentbank(results, eval_intermediates=False)
            for k, v in em.items():
                self.log(f"ExactMatch_{k}_{split}", v, on_step=False, on_epoch=True)
            for k, v in f1.items():
                self.log(f"F1_{k}_{split}", v, on_step=False, on_epoch=True)

        elif self.dataset == "ruletaker":
            answer_accuracies, proof_accuracies = evaluate_ruletaker(results)
            for k in answer_accuracies.keys():
                self.log(
                    f"Accuracy_answer_{k}_{split}",
                    answer_accuracies[k],
                    on_step=False,
                    on_epoch=True,
                )
                self.log(
                    f"Accuracy_proof_{k}_{split}",
                    proof_accuracies[k],
                    on_step=False,
                    on_epoch=True,
                )

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
 
