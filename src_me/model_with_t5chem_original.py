#!/usr/bin/env python
# coding: utf-8
# # Text2Smiles using Chemistry T5 model Model pretrained on pubchem with tgx data with zscore<2 and superior to 2

# ## mol tokenizer

# In[1]:


import importlib
import os
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Union

import torch
from torchtext.vocab import Vocab
from tqdm import tqdm
from transformers import PreTrainedTokenizer

is_selfies_available: bool = False
if importlib.util.find_spec("selfies"):
    from selfies import split_selfies
    is_selfies_available = True
pattern: str = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex: re.Pattern = re.compile(pattern)
TASK_PREFIX: List[str] = ['Yield:', 'Product:', 'Fill-Mask:', 'Classification:', 'Reagents:', 'Reactants:']


# In[2]:


class MolTokenizer(ABC, PreTrainedTokenizer):
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
        task_prefixs: List[str]=[],
        **kwargs
    ) -> None:
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            pad_token=pad_token,
            eos_token=eos_token,
            mask_token=mask_token,
            **kwargs)

        task_prefixs = TASK_PREFIX+task_prefixs
        self.create_vocab(
            vocab_file=vocab_file, 
            source_files=source_files, 
            vocab_size=max_size-len(task_prefixs)
            )
        if self.vocab:
            extra_to_add: int = max_size - len(self.vocab)
            cur_added_len: int = len(task_prefixs) + 9 # placeholder for smiles tokens
            for i in range(cur_added_len, extra_to_add):
                task_prefixs.append('<extra_task_{}>'.format(str(i)))
            self.add_tokens(['<extra_token_'+str(i)+'>' for i in range(9)]+task_prefixs+['>'], special_tokens=True)
            self.unique_no_split_tokens = sorted(
                set(self.unique_no_split_tokens).union(set(self.all_special_tokens))
            )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def merge_vocabs(
        self, 
        vocabs: List[Vocab], 
        vocab_size: Optional[int]=None,
    ) -> Vocab:
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
                    specials=special_tokens,
                    max_size=vocab_size-len(special_tokens) if vocab_size else vocab_size)

    def create_vocab(
        self, 
        vocab_file: Optional[str]=None,
        source_files: Optional[Union[str, List[str]]]=None,
        vocab_size: Optional[int]=None,
        ) -> None:
        """
        Create a vocabulary from current vocabulary file or from source file(s).
        Args:
            vocab_file (:obj:`string`, `optional`, defaults to ''):
                File containing the vocabulary (torchtext.vocab.Vocab class).
            source_files (:obj:`string`, `optional`, defaults to ''):
                File containing source data files, vocabulary would be built based on the source file(s).
            vocab_size: (:obj:`int`, `optional`, defaults to `None`):
                The final vocabulary size. `None` for no limit.
        """
        if vocab_file:
            if not os.path.isfile(vocab_file):
                raise ValueError(
                    "Can't find a vocabulary file at path '{}'.".format(vocab_file)
                )
            else:
                self.vocab: Vocab = self.merge_vocabs([torch.load(vocab_file)], vocab_size=vocab_size)

        elif source_files:
            if isinstance(source_files, str):
                if not os.path.isfile(source_files):
                    raise ValueError(
                        "Can't find a source file at path '{}'.".format(source_files)
                    )
                else:
                    source_files = [source_files]
            counter: Dict[int, Counter] = {}
            vocabs: Dict[int, Vocab] = {}
            for i, source_file in enumerate(source_files):
                counter[i] = Counter()
                with open(source_file) as rf:
                    for line in tqdm(rf, desc='Generating {}'.format(source_file)):
                        try:
                            items: List[str] = self._tokenize(line.strip())
                            counter[i].update(items)
                        except AssertionError:
                            print(line.strip())
                specials: List[str] = list(self.special_tokens_map.values()) # type: ignore
                vocabs[i] = Vocab(counter[i], specials=specials)
            self.vocab = self.merge_vocabs([vocabs[i] for i in range(len(source_files))], vocab_size=vocab_size)
        else:
            self.vocab = None

    def get_vocab(self) -> Dict[str, int]:
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab
    
    @abstractmethod
    def _tokenize(self, text: str, **kwargs) -> List[str]: 
        """
        Tokenize a molecule or reaction
        """
        pass

    def _convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str) in an id using the vocab. """
        assert isinstance(self.vocab, Vocab),\
            'No vocabulary found! Need to be generated at initialization or using .create_vocab method.'
        return self.vocab.stoi[token]

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        assert isinstance(self.vocab, Vocab),\
            'No vocabulary found! Need to be generated at initialization or using .create_vocab method.'
        return self.vocab.itos[index]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """ Converts a sequence of tokens (string) in a single string. """
        out_string: str = "".join(tokens).strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A Mol sequence has the following format:
        - single sequence: ``<s> X </s>``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

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


# In[3]:


class SimpleTokenizer(MolTokenizer):
    r"""
    Constructs a simple, character-level tokenizer. Based on SMILES.
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (:obj:`string`):
            File containing the vocabulary (torchtext.vocab.Vocab class).
        source_files (:obj:`string`, `optional`, defaults to ''):
            File containing source data files, vocabulary would be built based on the source file(s).
        unk_token (:obj:`string`, `optional`, defaults to '<unk>'):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (:obj:`string`, `optional`, defaults to '<s>'):
            string: a beginning of sentence token.
        pad_token (:obj:`string`, `optional`, defaults to "<pad>"):
            The token used for padding, for example when batching sequences of different lengths.
        eos_token (:obj:`string`, `optional`, defaults to '</s>'):
            string: an end of sentence token.
        max_size: (:obj:`int`, `optional`, defaults to 100):
            The final vocabulary size. `None` for no limit.
        **kwargs：
            Arguments passed to `~transformers.PreTrainedTokenizer`
    """
    def __init__(self, vocab_file, max_size=100, **kwargs) -> None:
        super().__init__(vocab_file=vocab_file, max_size=max_size, **kwargs)

    def _tokenize(self, text: str, **kwargs) -> List[str]: 
        return list(text)


# In[4]:


class AtomTokenizer(MolTokenizer):
    r"""
    Constructs an atom-level tokenizer. Based on SMILES.
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (:obj:`string`):
            File containing the vocabulary (torchtext.vocab.Vocab class).
        source_files (:obj:`string`, `optional`, defaults to ''):
            File containing source data files, vocabulary would be built based on the source file(s).
        unk_token (:obj:`string`, `optional`, defaults to '<unk>'):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (:obj:`string`, `optional`, defaults to '<s>'):
            string: a beginning of sentence token.
        pad_token (:obj:`string`, `optional`, defaults to "<pad>"):
            The token used for padding, for example when batching sequences of different lengths.
        eos_token (:obj:`string`, `optional`, defaults to '</s>'):
            string: an end of sentence token.
        max_size: (:obj:`int`, `optional`, defaults to 1000):
            The final vocabulary size. `None` for no limit.
        **kwargs：
            Arguments passed to `~transformers.PreTrainedTokenizer`
    """
    def __init__(self, vocab_file, max_size=1000, **kwargs) -> None:
        super().__init__(vocab_file=vocab_file, max_size=max_size, **kwargs)

    def _tokenize(self, text: str, **kwargs) -> List[str]: 
        tokens: List[str] = [token for token in regex.findall(text)]
        assert text == ''.join(tokens), 'Error when parsing {}'.format(text)
        return tokens


# In[5]:


class SelfiesTokenizer(MolTokenizer):
    r"""
    Constructs an SELFIES tokenizer. Based on SELFIES.
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (:obj:`string`):
            File containing the vocabulary (torchtext.vocab.Vocab class).
        source_files (:obj:`string`, `optional`, defaults to ''):
            File containing source data files, vocabulary would be built based on the source file(s).
        unk_token (:obj:`string`, `optional`, defaults to '<unk>'):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (:obj:`string`, `optional`, defaults to '<s>'):
            string: a beginning of sentence token.
        pad_token (:obj:`string`, `optional`, defaults to "<pad>"):
            The token used for padding, for example when batching sequences of different lengths.
        eos_token (:obj:`string`, `optional`, defaults to '</s>'):
            string: an end of sentence token.
        max_size: (:obj:`int`, `optional`, defaults to 1000):
            The final vocabulary size. `None` for no limit.
        **kwargs：
            Arguments passed to `~transformers.PreTrainedTokenizer`
    """
    def __init__(self, vocab_file, max_size=1000, **kwargs) -> None:
        super().__init__(vocab_file=vocab_file, max_size=max_size, **kwargs)
        assert is_selfies_available, "You need to install selfies package to use SelfiesTokenizer"

    def _tokenize(self, text: str, **kwargs) -> List[str]: 
        """
        Tokenize a SELFIES molecule or reaction
        """
        return list(split_selfies(text))


# ## data utils

# In[6]:


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


class TaskSettings(NamedTuple):
    prefix: str
    max_source_length: int
    max_target_length: int
    output_layer: str


T5ChemTasks: Dict[str, TaskSettings] = {
    'gex': TaskSettings('', 1000, 200, 'seq2seq'),
    'product': TaskSettings('Product:', 400, 200, 'seq2seq'),
    'reactants': TaskSettings('Reactants:', 200, 300, 'seq2seq'),
    'reagents': TaskSettings('Reagents:', 400, 200, 'seq2seq'),
    'classification': TaskSettings('Classification:', 500, 1, 'classification'),
    'regression': TaskSettings('Yield:', 500, 1, 'regression'),
    'pretrain': TaskSettings('Fill-Mask:', 400, 200, 'seq2seq'),
    'mixed': TaskSettings('', 400, 300, 'seq2seq'),
}


class LineByLineTextDataset(Dataset):
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer, 
        file_path: str, 
        block_size: int, 
        prefix: str = ''
    ) -> None:
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        
        self.prefix: str = prefix
        self._file_path: str = file_path
        self._len: int = int(subprocess.check_output("wc -l " + file_path, shell=True).split()[0])
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_length: int = block_size
        
    def __getitem__(self, idx: int) -> torch.Tensor:
        line: str = linecache.getline(self._file_path, idx + 1).strip()
        sample: BatchEncoding = self.tokenizer(
                        self.prefix+line,
                        max_length=self.max_length,
                        padding="do_not_pad",
                        truncation=True,
                        return_tensors='pt',
                    )
        return sample['input_ids'].squeeze(0)
      
    def __len__(self) -> int:
        return self._len


class TaskPrefixDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_dir: str,
        prefix: str='',
        type_path: str="train",
        max_source_length: int=300,
        max_target_length: int=100,
        separate_vocab: bool=False,
    ) -> None:
        super().__init__()

        self.prefix: str = prefix
        self._source_path: str = os.path.join(data_dir, type_path + ".source")
        self._target_path: str = os.path.join(data_dir, type_path + ".target")
        self._len_source: int = int(subprocess.check_output("wc -l " + self._source_path, shell=True).split()[0])
        self._len_target: int = int(subprocess.check_output("wc -l " + self._target_path, shell=True).split()[0])
        assert self._len_source == self._len_target, "Source file and target file don't match!"
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_source_len: int = max_source_length
        self.max_target_len: int = max_target_length
        self.sep_vocab: bool = separate_vocab

    def __len__(self) -> int:
        return self._len_source

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        source_line: str = linecache.getline(self._source_path, idx + 1).strip()
        source_sample: BatchEncoding = self.tokenizer(
                        self.prefix+source_line,
                        max_length=self.max_source_len,
                        padding="do_not_pad",
                        truncation=True,
                        return_tensors='pt',
                    )
        target_line: str = linecache.getline(self._target_path, idx + 1).strip()
        if self.sep_vocab:
            try:
                target_value: float = float(target_line)
                target_ids: torch.Tensor = torch.Tensor([target_value])
            except TypeError:
                print("The target should be a number, \
                        not {}".format(target_line))
                raise AssertionError
        else:
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
        return {"input_ids": source_ids, "attention_mask": src_mask,
                "decoder_input_ids": target_ids}

    def sort_key(self, ex: BatchEncoding) -> int:
        """ Sort using length of source sentences. """
        return len(ex['input_ids'])


def data_collator(batch: List[BatchEncoding], pad_token_id: int) -> Dict[str, torch.Tensor]:
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


def CalMSELoss(model_output: PredictionOutput) -> Dict[str, float]:
    predictions: np.ndarray = model_output.predictions # type: ignore
    label_ids: np.ndarray = model_output.label_ids.squeeze() # type: ignore
    loss: float = ((predictions - label_ids)**2).mean().item()
    return {'mse_loss': loss}

def AccuracyMetrics(model_output: PredictionOutput) -> Dict[str, float]:
    label_ids: np.ndarray = model_output.label_ids # type: ignore
    predictions: np.ndarray = model_output.predictions.reshape(-1, label_ids.shape[1]) # type: ignore
    correct: int = np.all(predictions==label_ids, 1).sum()
    return {'accuracy': correct/len(predictions)}


# ## model

# In[ ]:





# ## trainer

# In[7]:


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


# In[ ]:





# ## run trainer

# In[8]:


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

#from .data_utils import (AccuracyMetrics, CalMSELoss, LineByLineTextDataset,
#                        T5ChemTasks, TaskPrefixDataset, TaskSettings,
#                        data_collator)
# from .model import T5ForProperty
# from .mol_tokenizers import (AtomTokenizer, MolTokenizer, SelfiesTokenizer,
#                            SimpleTokenizer)
# from .trainer import EarlyStopTrainer

tokenizer_map: Dict[str, MolTokenizer] = {
    'simple': SimpleTokenizer,  # type: ignore
    'atom': AtomTokenizer,  # type: ignore
    'selfies': SelfiesTokenizer,    # type: ignore
}


# ### arguments

# In[9]:


data_dir = '../Data/data_t5chem/MCF7_24h_10um'#'t5chem_model/data/data/sample/pretrain/'
output_dir = '../Results/model_t5chem'
task_type = 'gex'
pretrain = 't5chem_model/models/pretrain/simple'
vocab = ''
tokenizer = ''
random_seed = 8570
num_epoch = 100
log_step = 5000
batch_size = 32
init_lr = 5e-4
num_classes = None


# ### set cuda device

# In[10]:


torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
# this one is needed for torchtext random call (shuffled iterator)
# in multi gpu it ensures datasets are read in the same order
random.seed(random_seed)
# some cudnn methods can be random even after fixing the seed
# unless you tell it to be deterministic
torch.backends.cudnn.deterministic = True


# ### set tasks

# In[11]:


assert task_type in T5ChemTasks, \
    "only {} are currenly supported, but got {}".\
        format(tuple(T5ChemTasks.keys()), task_type)
task: TaskSettings = T5ChemTasks[task_type]


# ### load pretrain model

# In[12]:


model = T5ForConditionalGeneration.from_pretrained(pretrain)


# ### load pretrain tokenizer

# In[13]:


if not hasattr(model.config, 'tokenizer'):
    logging.warning("No tokenizer type detected, will use SimpleTokenizer as default")
tokenizer_type = getattr(model.config, "tokenizer", 'simple')
vocab_path = os.path.join(pretrain, 'vocab.pt')
if not os.path.isfile(vocab_path):
    vocab_path = vocab
    if not vocab_path:
        raise ValueError(
                "Can't find a vocabulary file at path '{}'.".format(pretrain)
            )
tokenizer = tokenizer_map[tokenizer_type](vocab_file=vocab_path)
model.config.tokenizer = tokenizer_type # type: ignore
model.config.task_type = task_type # type: ignore


# ### save vocab

# In[14]:


os.makedirs(output_dir, exist_ok=True)
tokenizer.save_vocabulary(os.path.join(output_dir, 'vocab.pt'))


# In[27]:


tokenizer.get_vocab()


# ### create dataset

# In[15]:

"""
# Training
# Create the dataset with specific tokens 
dataset = LineByLineTextDataset(
    tokenizer=tokenizer, # The tokenizer to be used
    file_path=os.path.join(data_dir,'train.txt'), # This is the path to the text file containing the dataset.
    block_size=task.max_source_length, # This specifies the maximum length of each tokenized sequence.
    prefix=task.prefix, #This is an optional prefix that will be prepended to each line of text read from the file.
)
# simplifies the preparation of data for language modeling tasks
data_collator_padded = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


# In[16]:


# Evaluation
do_eval = os.path.exists(os.path.join(data_dir, 'val.txt'))
if do_eval:
    eval_strategy = "steps"
    eval_iter = LineByLineTextDataset(
        tokenizer=tokenizer, 
        file_path=os.path.join(data_dir,'val.txt'),
        block_size=task.max_source_length,
        prefix=task.prefix,
    )
else:
    eval_strategy = "no"
    eval_iter = None
"""
# ### create dataset task prefix
# Training
dataset = TaskPrefixDataset(
    tokenizer, 
    data_dir=data_dir,
    prefix=task.prefix,
    max_source_length=task.max_source_length,
    max_target_length=task.max_target_length,
    separate_vocab=(task.output_layer != 'seq2seq'),
    type_path="train",
)
data_collator_padded = partial(
    data_collator, pad_token_id=tokenizer.pad_token_id)
# Evaluation
do_eval = os.path.exists(os.path.join(data_dir, 'val.source'))
if do_eval:
    eval_strategy = "steps"
    eval_iter = TaskPrefixDataset(
        tokenizer, 
        data_dir=data_dir,
        prefix=task.prefix,
        max_source_length=task.max_source_length,
        max_target_length=task.max_target_length,
        separate_vocab=(task.output_layer != 'seq2seq'),
        type_path="val",
    )
else:
    eval_strategy = "no"
    eval_iter = None
# ### metrics

# In[20]:


compute_metrics = None


# ### training arguments

# In[21]:


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


# In[22]:


trainer = EarlyStopTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator_padded,
    train_dataset=dataset,
    eval_dataset=eval_iter,
    compute_metrics=compute_metrics,
)


# In[23]:


trainer.train()


# In[24]:


print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))  # Change 0 to the GPU device index you want to check


# In[ ]:





# In[ ]:





# In[ ]:




