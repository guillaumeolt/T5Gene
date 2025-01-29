#!/usr/bin/env python
# coding: utf-8

# # GPU: # Text2Smiles using Chemistry T5 model Model pretrained on pubchem with tgx data with zscore<2 and superior to 2

# In[1]:


import importlib
import os
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torchtext.vocab import Vocab, build_vocab_from_iterator
import torchtext.vocab.vocab as torchtext_vocab
from tqdm import tqdm
from transformers import PreTrainedTokenizer

#is_selfies_available: bool = False
#if importlib.util.find_spec("selfies"):
#    from selfies import split_selfies
#    is_selfies_available = True
#pattern: str = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
#regex: re.Pattern = re.compile(pattern)
#TASK_PREFIX: List[str] = ['Yield:', 'Product:', 'Fill-Mask:', 'Classification:', 'Reagents:', 'Reactants:']


# In[2]:


print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))  # Change 0 to the GPU device index you want to check


# ## mol tokenizer

# In[29]:





# In[3]:


class MolAndGeneTokenizer(ABC, PreTrainedTokenizer):
    r"""
    An abstract class for all tokenizers. Other tokenizer should
    inherit this class
    """
    def __init__(
        self,
        vocab_file: Optional[str]=None,
        source_files: Optional[Union[str, List[str]]]=None,
        unk_token: str='<unk>',
        bos_token: str='<s>',
        pad_token: str="<pad>",
        eos_token: str='</s>',
        mask_token: str='<mask>',
        max_size: int=1000,
        **kwargs
    ) -> None:
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            pad_token=pad_token,
            eos_token=eos_token,
            mask_token=mask_token,
            **kwargs)
        self.create_vocab(
            vocab_file=vocab_file,
            )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def merge_vocabs(
        self, 
        vocabs: List[Vocab]
    ) -> Vocab: # type: ignore
        """
        Merge individual vocabularies (assumed to be generated from disjoint
        documents) into a larger vocabulary.
        Args:
            vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
            vocab_size: `int` the final vocabulary size. `None` for no limit.
        Return:
            `torchtext.vocab.Vocab`
        """
        merged: Counter = sum([vocab.freqs for vocab in vocabs], Counter())
        special_tokens: List[str] = list(self.special_tokens_map.values())  # type: ignore
        return Vocab(merged,
                    specials=special_tokens)
    
    def read_vocab(self, path):
        vocab = dict()
        with open(path, 'r') as f:
            for line in f:
                index, token = line.strip().split('\t')
                vocab[token] = int(index)
        return vocab
    
    def create_vocab(
        self, 
        vocab_file: Optional[str]=None,
        ) -> None:
        """
        Create a vocabulary from current vocabulary txt file.
        """
        if vocab_file:
            if not os.path.isfile(vocab_file):
                raise ValueError(
                    "Can't find a vocabulary file at path '{}'.".format(vocab_file)
                )
            else:
                new_vocab = self.read_vocab(vocab_file)
                special_tokens: List[str] = list(self.special_tokens_map.values())  # type: ignore
                self.vocab = torchtext_vocab(new_vocab, specials=special_tokens)

    def get_vocab(self) -> Dict[str, int]:
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab
    
    @abstractmethod
    def _tokenize(self, text: str, **kwargs) -> List[str]: 
        """
        Tokenize
        """
        pass

    def _convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str) in an id using the vocab. """
        assert isinstance(self.vocab, Vocab),\
            'No vocabulary found! Need to be generated at initialization or using .create_vocab method.'
        return self.vocab.get_stoi()[token]

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        assert isinstance(self.vocab, Vocab),\
            'No vocabulary found! Need to be generated at initialization or using .create_vocab method.'
        return self.vocab.get_itos()[index]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """ Converts a sequence of tokens (string) in a single string. """
        out_string: str = "".join(tokens).strip()
        return out_string

    def save_vocabulary(self, vocab_path: str) -> None:    # type: ignore
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.
        Args:
            vocab_path (:obj:`str`):
                The directory in which to save the vocabulary.
        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        torch.save(self.vocab, vocab_path)


# In[4]:


class MyGeneTokenizer(MolAndGeneTokenizer):
    """
    Constructs a simple, character-level tokenizer. Based on GENE names.
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    """
    def __init__(self, vocab_file, max_size=100, **kwargs) -> None:
        super().__init__(vocab_file=vocab_file, max_size=max_size, **kwargs)
    def _tokenize(self, text: str, **kwargs) -> List[str]:
        # Implement your custom tokenization logic here
        return text.split()  # Example: simple whitespace tokenization


# ## data utils

# In[5]:


import linecache
import os
import subprocess
from typing import Dict, List, NamedTuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import BatchEncoding, PreTrainedTokenizer
from transformers.trainer_utils import PredictionOutput

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


# ## model

# In[ ]:





# ## trainer

# In[6]:


import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer
from transformers.trainer_pt_utils import (DistributedTensorGatherer,
                                           nested_concat)
from transformers.trainer_utils import EvalPrediction, PredictionOutput


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


# ## run trainer

# In[7]:


import argparse
import logging
import os
import random
from functools import partial
from typing import Dict

import numpy as np
import torch
from transformers import (DataCollatorForLanguageModeling, T5Config,
                          T5ForConditionalGeneration, TrainingArguments)


# ### arguments

# In[8]:


data_dir = '../Data/data_t5chem/MCF7_24h_10um/'#'t5chem_model/data/data/sample/pretrain/'
output_dir = '../Results/model_t5chem_gene'
pretrain = 't5chem_model/models/pretrain/simple'
vocab = ''
tokenizer = ''
max_source_length=1000
max_target_length=200
output_layer = 'seq2seq'
random_seed = 8570
num_epoch = 100
log_step = 5000
batch_size = 32
init_lr = 5e-4
num_classes = None


# ### set cuda device

# In[9]:


torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
# this one is needed for torchtext random call (shuffled iterator)
# in multi gpu it ensures datasets are read in the same order
random.seed(random_seed)
# some cudnn methods can be random even after fixing the seed
# unless you tell it to be deterministic
torch.backends.cudnn.deterministic = True


# ### load pretrain tokenizer

# In[12]:


vocab_path = "../Data/data_t5chem/MCF7_24h_10um/vocab.txt"
tokenizer = MyGeneTokenizer("../Data/data_t5chem/MCF7_24h_10um/vocab.txt")


# ### load pretrain model

# In[13]:


model = T5ForConditionalGeneration.from_pretrained(pretrain)


# In[14]:


# change embedding layer
new_vocab_size = tokenizer.vocab_size
embedding_dim = model.shared.weight.size(1)  # Keep the same embedding dimension

# Reinitialize the embedding layer
model.shared = nn.Embedding(new_vocab_size, embedding_dim)

# If you have tied embeddings (e.g., in T5), you might need to update other parts as well
model.encoder.embed_tokens = model.shared
model.decoder.embed_tokens = model.shared

# If using tied weights, you need to tie them again
model.lm_head.weight = model.shared.weight


# ### save vocab

# In[15]:


os.makedirs(output_dir, exist_ok=True)
tokenizer.save_vocabulary(os.path.join(output_dir, 'vocab.pt'))


# ### create dataset

# In[16]:


dataset = TaskDataset(
    tokenizer, 
    data_dir=data_dir,
    max_source_length=max_source_length,
    max_target_length=max_target_length,
    type_path="train",
)
data_collator_padded = partial(
    data_collator, pad_token_id=tokenizer.vocab.__getitem__(tokenizer.pad_token))


# In[17]:


do_eval = os.path.exists(os.path.join(data_dir, 'val.source'))
if do_eval:
    eval_strategy = "steps"
    eval_iter = TaskDataset(
        tokenizer, 
        data_dir=data_dir,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        type_path="val",
    )
else:
    eval_strategy = "no"
    eval_iter = None


# ### metrics

# In[18]:


compute_metrics = None


# ### training arguments

# In[19]:


training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    do_train=True,
    evaluation_strategy=eval_strategy,
    num_train_epochs=num_epoch,
    per_device_train_batch_size=batch_size,
    logging_steps=log_step,
    per_device_eval_batch_size=batch_size,
    save_steps=10000,
    save_total_limit=5,
    learning_rate=init_lr,
    prediction_loss_only=(compute_metrics is None),
)


# In[20]:


trainer = EarlyStopTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator_padded,
    train_dataset=dataset,
    eval_dataset=eval_iter,
    compute_metrics=compute_metrics,
)


# In[21]:


trainer.train()



