from dataclasses import dataclass, field

from data import IndexingTrainDataset, IndexingCollator, QueryEvalCollator
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, TrainerCallback
from trainer import IndexingTrainer
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser

SPIECE_UNDERLINE = "▁"


@dataclass
class HyperParameters:
    model_name: str = field(default='t5-large')
    max_length: int = field(
        default=32,
        metadata={"help": "only use the first 32 tokens of documents (including title)"})
    # paths
    path_to_traindata: str = field(default='data/NQ/NQ_10k_multi_task_train.json')
    path_to_testdata: str = field(default='data/NQ/NQ_10k_valid.json')
    cache_dir: str = field(default='cache')
    output_dir = "results"
    # wandb logging
    project_name: str = "DSI"
    run_name: str = 'NQ-10k-t5-large'


@dataclass
class DSITrainArgs:
    # training args
    learning_rate: float = 5e-4
    warmup_steps: int = 10000
    weight_decay: float = field(default=0, metadata={"help": "typically, set this to 0.01"})
    per_device_train_batch_size: int = 128
    per_device_eval_batch_size: int = 128
    evaluation_strategy: str = 'steps'
    eval_steps: int = 1000
    max_steps: int = 1000000
    dataloader_drop_last: bool = False
    report_to: str = 'wandb'
    logging_steps: int = 50
    save_strategy: str = 'no'
    fp16: bool = field(
        default=False,
        metadata={
            "help":
                "gives 0/nan loss at some point during training, seems this is a transformers bug."
        })
    dataloader_num_workers: int = 10
    gradient_accumulation_steps: int = 1


class QueryEvalCallback(TrainerCallback):

    def __init__(self, test_dataset, logger, restrict_decode_vocab, args: TrainingArguments,
                 tokenizer: T5Tokenizer):
        self.tokenizer = tokenizer
        self.logger = logger
        self.args = args
        self.test_dataset = test_dataset
        self.restrict_decode_vocab = restrict_decode_vocab
        self.dataloader = DataLoader(
            test_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=QueryEvalCollator(self.tokenizer, padding='longest'),
            shuffle=False,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        hit_at_1 = 0
        hit_at_10 = 0
        model = kwargs['model'].eval()
        for batch in tqdm(self.dataloader, desc='Evaluating dev queries'):
            inputs, labels = batch
            with torch.no_grad():
                batch_beams = model.generate(
                    inputs['input_ids'].to(model.device),
                    max_length=20,
                    num_beams=10,
                    prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                    num_return_sequences=10,
                    early_stopping=True,
                ).reshape(inputs['input_ids'].shape[0], 10, -1)
                for beams, label in zip(batch_beams, labels):
                    rank_list = self.tokenizer.batch_decode(
                        beams, skip_special_tokens=True
                    )  # beam search should not return repeated docids but somehow due to T5 tokenizer there some repeats.
                    hits = np.where(np.array(rank_list)[:10] == label)[0]
                    if len(hits) != 0:
                        hit_at_10 += 1
                        if hits[0] == 0:
                            hit_at_1 += 1
        self.logger.log({
            "Hits@1": hit_at_1 / len(self.test_dataset),
            "Hits@10": hit_at_10 / len(self.test_dataset)
        })


def compute_metrics(eval_preds):
    num_predict = 0
    num_correct = 0
    for predict, label in zip(eval_preds.predictions, eval_preds.label_ids):
        num_predict += 1
        if len(np.where(predict == 1)[0]) == 0:
            continue
        if np.array_equal(
                label[:np.where(label == 1)[0].item()],
                predict[np.where(predict == 0)[0][0].item() + 1:np.where(predict == 1)[0].item()]):
            num_correct += 1

    return {'accuracy': num_correct / num_predict}


def main():
    parser = HfArgumentParser([HyperParameters, DSITrainArgs])
    hp, dsi_train_args = parser.parse_args_into_dataclasses()

    # We use wandb to log Hits scores after each epoch. Note, this script does not save model checkpoints.
    wandb.login()
    wandb.init(project=hp.project_name, name=hp.run_name)

    tokenizer = T5Tokenizer.from_pretrained(hp.model_name, cache_dir=hp.cache_dir)
    model = T5ForConditionalGeneration.from_pretrained(hp.model_name, cache_dir=hp.cache_dir)

    train_dataset = IndexingTrainDataset(path_to_data=hp.path_to_traindata,
                                         max_length=hp.max_length,
                                         cache_dir=hp.cache_dir,
                                         tokenizer=tokenizer)

    # This eval set is really not the 'eval' set but used to report if the model can memorise (index) all training data points.
    eval_dataset = IndexingTrainDataset(path_to_data=hp.path_to_traindata,
                                        max_length=hp.max_length,
                                        cache_dir=hp.cache_dir,
                                        tokenizer=tokenizer)

    # This is the actual eval set.
    test_dataset = IndexingTrainDataset(path_to_data=hp.path_to_testdata,
                                        max_length=hp.max_length,
                                        cache_dir=hp.cache_dir,
                                        tokenizer=tokenizer)

    ################################################################
    # docid generation constrain, we only generate integer docids.
    INT_TOKEN_IDS = []
    for token, id in tokenizer.get_vocab().items():
        if token[0] == SPIECE_UNDERLINE:
            if token[1:].isdigit():
                INT_TOKEN_IDS.append(id)
        if token == SPIECE_UNDERLINE:
            INT_TOKEN_IDS.append(id)
        elif token.isdigit():
            INT_TOKEN_IDS.append(id)
    INT_TOKEN_IDS.append(tokenizer.eos_token_id)

    def restrict_decode_vocab(batch_idx, prefix_beam):
        return INT_TOKEN_IDS

    ################################################################

    training_args = TrainingArguments(**dsi_train_args.__dict__)

    trainer = IndexingTrainer(model=model,
                              tokenizer=tokenizer,
                              args=training_args,
                              train_dataset=train_dataset,
                              eval_dataset=eval_dataset,
                              data_collator=IndexingCollator(
                                  tokenizer,
                                  padding='longest',
                              ),
                              compute_metrics=compute_metrics,
                              callbacks=[
                                  QueryEvalCallback(test_dataset, wandb, restrict_decode_vocab,
                                                    training_args, tokenizer)
                              ],
                              restrict_decode_vocab=restrict_decode_vocab)
    trainer.train()


if __name__ == "__main__":
    main()
