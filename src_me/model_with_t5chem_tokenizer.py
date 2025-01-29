#!/usr/bin/env python
# coding: utf-8
import importlib
import os
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Union

import torch
from torchtext.vocab import Vocab
import torchtext.vocab.vocab as torchtext_vocab
from transformers import PreTrainedTokenizer


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
