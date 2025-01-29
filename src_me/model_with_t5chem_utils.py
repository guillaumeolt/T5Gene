#!/usr/bin/env python
# coding: utf-8

import os
import shutil
from pathlib import Path
import linecache
import subprocess
from typing import Dict, List, Optional
from functools import partial

import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import Trainer
from transformers.trainer_pt_utils import (DistributedTensorGatherer,
                                           nested_concat)
from transformers.trainer_utils import EvalPrediction, PredictionOutput
from transformers.training_args import TrainingArguments
from transformers import BatchEncoding, PreTrainedTokenizer

from transformers import T5ForConditionalGeneration


class TaskDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_dir: str,
        type_path: str="train",
        max_source_length: int=300,
        max_target_length: int=100,
    ) -> None:
        super().__init__()

        self._source_path: str = os.path.join(data_dir, type_path + ".source")
        self._target_path: str = os.path.join(data_dir, type_path + ".target")
        self._len_source: int = int(subprocess.check_output("wc -l " + self._source_path, shell=True).split()[0])
        self._len_target: int = int(subprocess.check_output("wc -l " + self._target_path, shell=True).split()[0])
        assert self._len_source == self._len_target, "Source file and target file don't match!"
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_source_len: int = max_source_length
        self.max_target_len: int = max_target_length

    def __len__(self) -> int:
        return self._len_source

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        source_line: str = linecache.getline(self._source_path, idx + 1).strip()
        source_sample: BatchEncoding = self.tokenizer(
                        source_line,
                        max_length=self.max_source_len,
                        padding="do_not_pad",
                        truncation=True,
                        return_tensors='pt',
                    )
        target_line: str = linecache.getline(self._target_path, idx + 1).strip()
        target_line = ' '.join(target_line) # add space for my tokenizer
        target_sample: BatchEncoding = self.tokenizer(
                        target_line,
                        max_length=self.max_target_len,
                        padding="do_not_pad",
                        truncation=True,
                        return_tensors='pt',
                    )
        target_ids = target_sample["input_ids"].squeeze(0)
        source_ids: torch.Tensor = source_sample["input_ids"].squeeze(0)
        src_mask: torch.Tensor = source_sample["attention_mask"].squeeze(0)

        assert torch.all(source_ids < self.tokenizer.vocab_size), f"Invalid token ID found: {source_ids}"
        assert torch.all(target_ids < self.tokenizer.vocab_size), f"Invalid token ID found: {target_ids}"
        return {"input_ids": source_ids, "attention_mask": src_mask,
                "decoder_input_ids": target_ids}

    def sort_key(self, ex: BatchEncoding) -> int:
        """ Sort using length of source sentences. """
        return len(ex['input_ids'])


def data_collator(batch: List[BatchEncoding], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """
    The data_collator function processes a list of tokenized inputs (a batch), pads them to the same length using appropriate padding values,
    and organizes them into a dictionary suitable for model input. This function ensures that variable-length sequences are properly handled,
    allowing for efficient batching and input preparation for NLP models.
    """
    whole_batch: Dict[str, torch.Tensor] = {}
    ex: BatchEncoding = batch[0]
    for key in ex.keys():
        if 'mask' in key:
            padding_value = 0
        else:
            padding_value = pad_token_id
        whole_batch[key] = pad_sequence([x[key] for x in batch],
                                        batch_first=True,
                                        padding_value=padding_value)
    source_ids, source_mask, y = \
        whole_batch["input_ids"], whole_batch["attention_mask"], whole_batch["decoder_input_ids"]
    return {'input_ids': source_ids, 'attention_mask': source_mask,
            'labels': y}


class TaskLincsDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_dir: str,
        type_path: str="train",
        max_target_length: int=100,
    ) -> None:
        super().__init__()

        self._source_path: str = os.path.join(data_dir, type_path + "_source.csv")
        self._target_path: str = os.path.join(data_dir, type_path + "_target.csv")
        self._len_source: int = int(subprocess.check_output("wc -l " + self._source_path, shell=True).split()[0])
        self._len_target: int = int(subprocess.check_output("wc -l " + self._target_path, shell=True).split()[0])
        assert self._len_source == self._len_target, "Source file and target file don't match!"
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_target_len: int = max_target_length

    def __len__(self) -> int:
        return self._len_source

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        source_line: str = linecache.getline(self._source_path, idx + 1).strip()
        source_line: str = linecache.getline(self._source_path, idx + 1).strip()
        source_sample: BatchEncoding =  torch.tensor([float(x) for x in source_line.split()[:]])#tensor
        
        target_line: str = linecache.getline(self._target_path, idx + 1).strip()
        target_line = ' '.join(target_line) # add space for my tokenizer
        target_sample: BatchEncoding = self.tokenizer(
                        target_line,
                        max_length=self.max_target_len,
                        padding="do_not_pad",
                        truncation=True,
                        return_tensors='pt',
                    )
        target_ids = target_sample["input_ids"].squeeze(0)

        source_embeds: torch.Tensor = source_sample.squeeze(0)
        src_mask: torch.Tensor = torch.ones_like(source_sample).squeeze(0)

        #assert torch.all(source_ids < self.tokenizer.vocab_size), f"Invalid token ID found: {source_ids}"
        assert torch.all(target_ids < self.tokenizer.vocab_size), f"Invalid token ID found: {target_ids}"
        return {"inputs_embeds": source_embeds, "attention_mask": src_mask,
                "decoder_input_ids": target_ids}

    def sort_key(self, ex: BatchEncoding) -> int:
        """ Sort using length of source sentences. """
        return len(ex['input_ids'])
def data_collator_lincs(batch: List[BatchEncoding], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """
    The data_collator function processes a list of tokenized inputs (a batch), pads them to the same length using appropriate padding values,
    and organizes them into a dictionary suitable for model input. This function ensures that variable-length sequences are properly handled,
    allowing for efficient batching and input preparation for NLP models.
    """
    whole_batch: Dict[str, torch.Tensor] = {}
    ex: BatchEncoding = batch[0]
    for key in ex.keys():
        if 'mask' in key:
            padding_value = 0
        else:
            padding_value = pad_token_id
        whole_batch[key] = pad_sequence([x[key] for x in batch],
                                        batch_first=True,
                                        padding_value=padding_value)
    source_mask, y = whole_batch["attention_mask"], whole_batch["decoder_input_ids"]

    inputs_embeds = pad_sequence([x["inputs_embeds"] for x in batch],batch_first=True)
    return {"inputs_embeds":inputs_embeds, 'attention_mask': source_mask, #'input_ids': source_ids, 'attention_mask': source_mask,
            'labels': y}


class EarlyStopTrainer(Trainer):
    """
    Save model weights based on validation error.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.min_eval_loss: float = float('inf')

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        eval_dataloader: DataLoader = self.get_eval_dataloader(eval_dataset)
        output: PredictionOutput = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        self.log(output.metrics) # type: ignore
        cur_loss: float = output.metrics['eval_loss'] # type: ignore
        if self.min_eval_loss >= cur_loss:
            self.min_eval_loss = cur_loss
            for f in Path(self.args.output_dir).glob('best_cp-*'):
                shutil.rmtree(f)
            output_dir: str = os.path.join(self.args.output_dir, f"best_cp-{self.state.global_step}")
            self.save_model(output_dir)
        return output.metrics # type: ignore

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.
        Works both with or without labels.
        """
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        model = self.model

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        losses_host = None
        preds_host = None
        labels_host = None

        world_size = 1

        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
        if not prediction_loss_only:
            preds_gatherer = DistributedTensorGatherer(world_size, num_examples)
            labels_gatherer = DistributedTensorGatherer(world_size, num_examples)

        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        for step, inputs in enumerate(dataloader):
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            if loss is not None:
                losses = loss.repeat(batch_size) # type: ignore
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0) # type: ignore
            if logits is not None:
                # preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
                logits = logits[0]
                logits_reduced = torch.argmax(logits, dim=-1) if (len(logits.size())>1 and logits.size()[-1]>2) else logits
                preds_host = logits_reduced if preds_host is None else nested_concat(preds_host, logits_reduced, padding_index=-100)
            if labels is not None:
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
                if not prediction_loss_only:
                    preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
                    labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
        if not prediction_loss_only:
            preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
            labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))

        eval_loss = eval_losses_gatherer.finalize()
        preds = preds_gatherer.finalize() if not prediction_loss_only else None
        label_ids = labels_gatherer.finalize() if not prediction_loss_only else None

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}

        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

