# LLM-proof
Repo for natural language proof generation with large language model with contrastive stepwise decoding

## Analysis of Prompting
Scripts for Vanilla Prompt, COT, and Select-and-Inference are placed under folder `./scripts`.
```
python prompting.py
python cot.py
python SI.py
```
## ConDec
ConDec is the framework of contrastive decoding with hard negatives. After finetuning with MLE loss, the generator is further adjusted with hard negatives.
For finetuning with MLE loss:
```
cd ./stepwise
CUDA_VISIBLE_DEVICES=0 python main.py fit --config cli_task1_stepwise_flan-t5-large.yaml
CUDA_VISIBLE_DEVICES=0 python main.py fit --config cli_task2_stepwise_flan-t5-large.yaml
```
### Training with Vanilla Hard Negatives
The vanilla hard negatives are constructed by randomly substituting the intermediate nodes with premises. For finetuning with vanilla hard neagtives:
```
cd ./ConDec
CUDA_VISIBLE_DEVICES=0 python main.py fit --config cli_task1_vanilla_flan-t5-large.yaml \
    --ckpt_path ../stepwise/ckpt_entailmentbank_task1/lightning_logs/version0/epoch\=499-step\=10500.ckpt
CUDA_VISIBLE_DEVICES=0 python main.py fit --config cli_task2_vanilla_flan-t5-large.yaml \
    --ckpt_path ../stepwise/ckpt_entailmentbank_task1/lightning_logs/version0/epoch\=599-step\=12600.ckpt
```
### Enhanced Hard Negatives Contruction
The construction of enhanced hard negatives consists of three stages: training the reasoner, inference with reasoner, and filtering with checker.
#### 1) Training the Reasoner
```
cd ./reasoner
```
Preprocess and sample the training data from training dataset
```
cd ./data_sample
python datasample.py
```
Since the gold proof tree for task1 and task2 are the same, the acquired training data is same either from task1 or task2. The reasoner is trained with the training data:
```
cd ./train
CUDA_VISIBLE_DEVICES=0 python main.py fit --config cli_entailmentbank_task1.yaml
```
#### 2) Inference with Reasoner
```
cd ./inference
CUDA_VISIBLE_DEVICES=0 python main.py validate --config cli_entailmentbank_task1.yaml
CUDA_VISIBLE_DEVICES=0 python main.py validate --config cli_entailmentbank_task2.yaml
```
This will result hard negatives for task1 and task2 seperately.
#### 3) Filtering with Checker
```
cd ./filter
python verify.py --task task1
```
Same with task2.

### Training with both Vanilla and Enhanced Hard Negatives
```
cd ./ConDec
CUDA_VISIBLE_DEVICES=0 python main.py fit --config cli_task1_enhanced_flan-t5-large.yaml \
    --ckpt_path ../stepwise/ckpt_entailmentbank_task1/lightning_logs/version0/epoch\=499-step\=10500.ckpt
CUDA_VISIBLE_DEVICES=0 python main.py fit --config cli_task2_enhanced_flan-t5-large.yaml \
    --ckpt_path ../stepwise/ckpt_entailmentbank_task1/lightning_logs/version0/epoch\=599-step\=12600.ckpt
```

### Evaluation
Evaluation refer to [official toolkit](https://github.com/allenai/entailment_bank).
