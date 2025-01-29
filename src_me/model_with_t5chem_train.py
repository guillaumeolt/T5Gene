#!/usr/bin/env python
# coding: utf-8
import argparse
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

from model_with_t5chem_tokenizer import MyGeneTokenizer
from model_with_t5chem_model import T5GeneToMol
from model_with_t5chem_utils import TaskDataset, data_collator, EarlyStopTrainer, TaskLincsDataset, data_collator_lincs

def train_lincs_text_mols():
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

    vocab_path = "../Data/data_t5chem/MCF7_24h_10um/vocab.txt"
    tokenizer = MyGeneTokenizer("../Data/data_t5chem/MCF7_24h_10um/vocab.txt")

    # ### load pretrain model

    model = T5ForConditionalGeneration.from_pretrained(pretrain)


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

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_vocabulary(os.path.join(output_dir, 'vocab.pt'))


    # ### create dataset

    dataset = TaskDataset(
        tokenizer, 
        data_dir=data_dir,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        type_path="train",
    )
    data_collator_padded = partial(
        data_collator, pad_token_id=tokenizer.vocab.__getitem__(tokenizer.pad_token))

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

    compute_metrics = None


    # ### training arguments

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

    trainer = EarlyStopTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator_padded,
        train_dataset=dataset,
        eval_dataset=eval_iter,
        compute_metrics=compute_metrics,
    )

    trainer.train()

def train_lincs_mols():

    data_dir = '../Data/data_t5chem/MCF7_24h_10um/'#'t5chem_model/data/data/sample/pretrain/'
    data_dir_lincs_mols = '../Data/data_t5chem/datasets_MCF7/lincs'
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

    vocab_path = "../Data/data_t5chem/MCF7_24h_10um/vocab.txt"
    tokenizer = MyGeneTokenizer("../Data/data_t5chem/MCF7_24h_10um/vocab.txt")

    # ### load pretrain model
    NEW_EMBEDDING_SIZE = 978
    model = T5GeneToMol.from_pretrained(pretrain, new_input_size=NEW_EMBEDDING_SIZE)

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

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_vocabulary(os.path.join(output_dir, 'vocab.pt'))


    # ### create dataset

    dataset = TaskLincsDataset(
        tokenizer=tokenizer, 
        data_dir=data_dir_lincs_mols,
        type_path="train",
        max_target_length=max_target_length,
    )
    data_collator_padded = partial(
        data_collator_lincs, pad_token_id=tokenizer.vocab.__getitem__(tokenizer.pad_token))

    do_eval = os.path.exists(os.path.join(data_dir_lincs_mols, 'val_source.csv'))
    if do_eval:
        eval_strategy = "steps"
        eval_iter = TaskLincsDataset(
            tokenizer, 
            data_dir=data_dir_lincs_mols,
            max_target_length=max_target_length,
            type_path="val",
        )
    else:
        eval_strategy = "no"
        eval_iter = None
    ## TEST
    if False:
        dataset = TaskDataset(
            tokenizer, 
            data_dir=data_dir,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            type_path="train",
        )
        data_collator_padded = partial(
            data_collator, pad_token_id=tokenizer.vocab.__getitem__(tokenizer.pad_token))

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

    compute_metrics = None


    # ### training arguments

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

    trainer = EarlyStopTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator_padded,
        train_dataset=dataset,
        eval_dataset=eval_iter,
        compute_metrics=compute_metrics,
    )

    trainer.train()

def train_lincs_frogs_mols():
    data_dir = '../Data/data_t5chem/MCF7_24h_10um/'#'t5chem_model/data/data/sample/pretrain/'
    data_dir_lincs_mols = '../Data/data_t5chem/datasets_MCF7/lincs_frogs'
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

    vocab_path = "../Data/data_t5chem/MCF7_24h_10um/vocab.txt"
    tokenizer = MyGeneTokenizer("../Data/data_t5chem/MCF7_24h_10um/vocab.txt")

    # ### load pretrain model
    NEW_EMBEDDING_SIZE = 512
    model = T5GeneToMol.from_pretrained(pretrain, new_input_size=NEW_EMBEDDING_SIZE)

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

    print(model)
    # ### save vocab

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_vocabulary(os.path.join(output_dir, 'vocab.pt'))


    # ### create dataset

    dataset = TaskLincsDataset(
        tokenizer=tokenizer, 
        data_dir=data_dir_lincs_mols,
        type_path="train",
        max_target_length=max_target_length,
    )
    data_collator_padded = partial(
        data_collator_lincs, pad_token_id=tokenizer.vocab.__getitem__(tokenizer.pad_token))

    do_eval = os.path.exists(os.path.join(data_dir_lincs_mols, 'val_source.csv'))
    if do_eval:
        eval_strategy = "steps"
        eval_iter = TaskLincsDataset(
            tokenizer, 
            data_dir=data_dir_lincs_mols,
            max_target_length=max_target_length,
            type_path="val",
        )
    else:
        eval_strategy = "no"
        eval_iter = None
    ## TEST
    if False:
        dataset = TaskDataset(
            tokenizer, 
            data_dir=data_dir,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            type_path="train",
        )
        data_collator_padded = partial(
            data_collator, pad_token_id=tokenizer.vocab.__getitem__(tokenizer.pad_token))

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

    compute_metrics = None


    # ### training arguments

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

    trainer = EarlyStopTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator_padded,
        train_dataset=dataset,
        eval_dataset=eval_iter,
        compute_metrics=compute_metrics,
    )

    trainer.train()

def add_args(parser):
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../Data/data_t5chem/datasets_MCF7/lincs_frogs",
        required=True,
        help="The input data dir.",
    )    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../Results/model_t5chem_gene",
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--pretrain",
        default='t5chem_model/models/pretrain/simple',
        help="Path to a pretrained model. If not given, we will train from scratch",
    )
    parser.add_argument(
        "--vocab",
        default="../Data/data_t5chem/MCF7_24h_10um/vocab.txt",
        help="Vocabulary file to load.",
    )
    parser.add_argument(
        "--random_seed",
        default=850,
        type=int,
        help="The random seed for model initialization",
    )
    parser.add_argument(
        "--num_epoch",
        default=100,
        type=int,
        help="Number of epochs for training.",
    )
    parser.add_argument(
        "--log_step",
        default=5000,
        type=int,
        help="Logging after every log_step",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--init_lr",
        default=5e-4,
        type=float,
        help="The initial learning rate for model training",
    )
    parser.add_argument(
        "--max_source_length",
        default=1000,
        type=int,
        help="The maximum length of the input source text",
    )
    parser.add_argument(
        "--max_target_length",
        default=200,
        type=int,
        help="The maximum length of the target text",
    )
    parser.add_argument(
        "--embedding_size",
        default=512,
        type=int,
        help="The size of the embedding layer",
    )
    parser.add_argument(
        "--method",
        default="lincs_frogs_mols",
        type=str,
        help="The method to train the model. Options are lincs_frogs_mols, lincs_mols, lincs_text_mols",
    )
# ### arguments

if __name__ == "__main__":

    #data_dir = '../Data/data_t5chem/MCF7_24h_10um/'#'t5chem_model/data/data/sample/pretrain/'
    #data_dir_lincs_mols = '../Data/data_t5chem/datasets_MCF7/lincs_frogs'
    #output_dir = '../Results/model_t5chem_gene'
    #pretrain = 't5chem_model/models/pretrain/simple'
    #vocab = ''
    #tokenizer = ''
    #max_source_length=1000
    #max_target_length=200
    #output_layer = 'seq2seq'
    #random_seed = 8570
    #num_epoch = 100
    #log_step = 5000
    #batch_size = 32
    #init_lr = 5e-4
    #num_classes = None

    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    # ### set cuda device

    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    # this one is needed for torchtext random call (shuffled iterator)
    # in multi gpu it ensures datasets are read in the same order
    random.seed(args.random_seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


    # ### load pretrain tokenizer

    tokenizer = MyGeneTokenizer(args.vocab)

    # ### load pretrain model
    if args.method == "lincs_frogs_mols":
        model = T5GeneToMol.from_pretrained(args.pretrain, new_input_size=args.embedding_size)
    if args.method == "lincs_mols":
        model = T5GeneToMol.from_pretrained(args.pretrain, new_input_size=args.embedding_size)
    if args.method == "lincs_text_mols":
        model = T5ForConditionalGeneration.from_pretrained(args.pretrain)

    # change embedding layer
    embedding_dim = model.shared.weight.size(1)  # Keep the same embedding dimension

    # Reinitialize the embedding layer
    model.shared = nn.Embedding(tokenizer.vocab_size, embedding_dim)

    # If you have tied embeddings (e.g., in T5), you might need to update other parts as well
    model.encoder.embed_tokens = model.shared
    model.decoder.embed_tokens = model.shared

    # If using tied weights, you need to tie them again
    model.lm_head.weight = model.shared.weight

    print(model)

    # ### save vocab

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_vocabulary(os.path.join(args.output_dir, 'vocab.pt'))

    # ### create dataset

    if args.method == "lincs_frogs_mols" or args.method == "lincs_mols":
        dataset = TaskLincsDataset(
            tokenizer=tokenizer, 
            data_dir=args.data_dir,
            type_path="train",
            max_target_length=args.max_target_length,
        )
        data_collator_padded = partial(
            data_collator_lincs, pad_token_id=tokenizer.vocab.__getitem__(tokenizer.pad_token))

        do_eval = os.path.exists(os.path.join(args.data_dir, 'val_source.csv'))
        if do_eval:
            eval_strategy = "steps"
            eval_iter = TaskLincsDataset(
                tokenizer, 
                data_dir=args.data_dir,
                max_target_length=args.max_target_length,
                type_path="val",
            )
        else:
            eval_strategy = "no"
            eval_iter = None
    ## TEST
    if args.method == "lincs_text_mols":
        dataset = TaskDataset(
            tokenizer, 
            data_dir=args.data_dir,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            type_path="train",
        )
        data_collator_padded = partial(
            data_collator, pad_token_id=tokenizer.vocab.__getitem__(tokenizer.pad_token))

        do_eval = os.path.exists(os.path.join(args.data_dir, 'val.source'))
        if do_eval:
            eval_strategy = "steps"
            eval_iter = TaskDataset(
                tokenizer, 
                data_dir=args.data_dir,
                max_source_length=args.max_source_length,
                max_target_length=args.max_target_length,
                type_path="val",
            )
        else:
            eval_strategy = "no"
            eval_iter = None

    # ### metrics

    compute_metrics = None

    # ### training arguments

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        evaluation_strategy=eval_strategy,
        num_train_epochs=args.num_epoch,
        per_device_train_batch_size=args.batch_size,
        logging_steps=args.log_step,
        per_device_eval_batch_size=args.batch_size,
        save_steps=10000,
        save_total_limit=5,
        learning_rate=args.init_lr,
        prediction_loss_only=(compute_metrics is None),
    )

    trainer = EarlyStopTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator_padded,
        train_dataset=dataset,
        eval_dataset=eval_iter,
        compute_metrics=compute_metrics,
    )

    trainer.train()
