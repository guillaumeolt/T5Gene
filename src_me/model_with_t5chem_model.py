#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os
import random
from functools import partial
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import (DataCollatorForLanguageModeling, T5Config,
                          T5ForConditionalGeneration, TrainingArguments)


# My T5 model on data-to-mol
class T5GeneToMol(T5ForConditionalGeneration):
    r"""
    T5Chem Model with a `language modeling` head on top modified to a linear layer. 
    Args:
    """
    #_keys_to_ignore_on_load_missing: List[str] = [
        #r"encoder\.embed_tokens\.weight",
        #r"decoder\.embed_tokens\.weight"
        #r"lm_head\.0\.weight",
        #r"lm_head\.0\.bias",
    #]
    #_keys_to_ignore_on_load_unexpected: List[str] = [
        #r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
        #r"lm_head\.weight",
    #]
    def __init__(
        self, 
        config: T5Config,
        new_input_size: int,
        ) -> None:
        super().__init__(config)
        self.linear = nn.Linear(new_input_size, config.d_model)
        self.new_input_size = new_input_size
        self.init_weights()
    

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache # ???
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict # ???


        if inputs_embeds is not None:
            inputs_embeds = self.linear(inputs_embeds)
            inputs_embeds = inputs_embeds.unsqueeze(1)  # Shape: [32, 1, 256]
            # Repeat the tensor along the new dimension to get the desired shape
            inputs_embeds = inputs_embeds.repeat(1, self.new_input_size, 1)  # Shape: [32, 978, 256]
            print(inputs_embeds.shape, "inputs_embeds_shape_555")


        # encode the input
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids, # type: ignore
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask, # type: ignore
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        """
        # decode the output
        print(decoder_input_ids, "decoder_input_ids_1")
        if decoder_input_ids is not None: ## !! what is that ??
            decoder_input_ids = decoder_input_ids[:, -1:]
        if decoder_inputs_embeds is not None:
            decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]
        print(decoder_input_ids, "decoder_input_ids_2")
        """ 
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    def generate(self, 
                 input_ids=None,
                 attention_mask=None,
                 early_stopping=None,
                 inputs_embeds=None, 
                 max_length=None, 
                 num_beams=None, 
                 num_return_sequences=None, 
                 decoder_start_token_id=None,  **kwargs):

        if inputs_embeds is not None:
            inputs_embeds = self.linear(inputs_embeds)
            inputs_embeds = inputs_embeds.unsqueeze(1)  # Shape: [32, 1, 256]
            # Repeat the tensor along the new dimension to get the desired shape
            inputs_embeds = inputs_embeds.repeat(1, self.new_input_size, 1)  # Shape: [32, 978, 256]

        return super().generate( input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 early_stopping=early_stopping,
                                 inputs_embeds=inputs_embeds,
                                 max_length=max_length,
                                 num_beams=num_beams, 
                                 num_return_sequences=num_return_sequences,
                                 decoder_start_token_id=decoder_start_token_id, **kwargs)