from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaPreTrainedModel,
    LlamaConfig,
    LlamaRotaryEmbedding,
    LlamaAttention,
    LlamaMLP,
    LlamaDecoderLayer,
)

from typing import Callable, List, Optional, Tuple, Union

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack

from typing import Optional, Tuple
from .utils import gumbel_softmax, compute_causal_mask, FractalCache

from transformers.utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from transformers.utils import ModelOutput

import torch
from typing import Optional, Tuple
import copy
import json
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.generation.utils import GenerateBeamDecoderOnlyOutput, GenerateDecoderOnlyOutput

from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)

import os
import time
import math

from transformers import BitsAndBytesConfig

logger = logging.get_logger(__name__)

class FractalLlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.is_verify =False
        

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
        
    def set_quantized_layers(self, quantized_layers):
        self.quantized_layers = quantized_layers

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        fractal_key_values_list: Optional[List[Cache]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        prefix_len_list: Optional[List[int]] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
                

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
            
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        fidx = 0 # index of fractal_key_values_list
        
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                if isinstance(decoder_layer, FractalLlamaDecoderLayer):
                    layer_outputs = decoder_layer(
                        input_ids=input_ids,
                        hidden_states=hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        fractal_key_value=fractal_key_values_list[fidx] if fractal_key_values_list is not None else None,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        prefix_len_list=prefix_len_list,
                        fidx=fidx,
                        **flash_attn_kwargs,
                    )
                    
                    input_ids = layer_outputs[-1]
                    position_ids = layer_outputs[-2]
                    position_embeddings = layer_outputs[-3]
                    attention_mask = layer_outputs[1]
                    fidx += 1
                    
                else:
                    layer_outputs = decoder_layer(
                        hidden_states=hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **flash_attn_kwargs,
                    )
            
                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = FractalModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            fractal_key_values_list=fractal_key_values_list if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        
        return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                # Fix else past_seen_tokens + sequence_length + 1 => else past_seen_tokens + sequence_length
                else past_seen_tokens + sequence_length
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask
    
@dataclass
class FractalModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    fractal_key_values_list: Optional[List[Tuple[Tuple[torch.FloatTensor]]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...

GenerateOutput = Union[GenerateDecoderOnlyOutput, GenerateBeamDecoderOnlyOutput]

class FractalLlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config):
        super().__init__(config)
        self.model = FractalLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.verify_mode_func = set_verify_mode

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        fractal_key_values_list: Optional[Union[List[Cache], List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            fractal_key_values_list=fractal_key_values_list,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return FractalCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            fractal_key_values_list=outputs.fractal_key_values_list,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )        
        
    def compute_fractal_key_values(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        fractal_key_values_list: Optional[List[Cache]] = None,
        use_cache: bool = True,
        output_attentions: bool = False,
        prefix_len_list: Optional[List[int]] = None,
        return_dict: bool = True,
        **flash_attn_kwargs,
    ) -> List[Cache]:
        return self.model.compute_fractal_key_values(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            fractal_key_values_list=fractal_key_values_list,
            use_cache=use_cache,
            output_attentions=output_attentions,
            prefix_len_list=prefix_len_list,
            return_dict=return_dict,
            **flash_attn_kwargs
        )
        
            
        
@dataclass
class FractalCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    fractal_key_values_list: Optional[List[Tuple[Tuple[torch.FloatTensor]]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class FractalModule(torch.nn.Module):
    def __init__(self,
                 config,
                 embedding_layer: torch.Tensor,
                 shared_lm_head: torch.nn.Linear,
                 decoder_layers: torch.nn.ModuleList,
                 norm_layer: LlamaRMSNorm = None,
                 tokenizer=None,
                 draft_layer_indexes=None,
                 rotary_emb=None,
    ):
        super().__init__()
    
        self.lm_head = copy.deepcopy(shared_lm_head)
        self.lm_head_device = next(self.lm_head.parameters()).device
        
        self.fractal_embedding = copy.deepcopy(embedding_layer)
        self.fractal_embedding_device = next(self.fractal_embedding.parameters()).device
        self.fractal_norm = copy.deepcopy(norm_layer)
        self.rotary_emb = rotary_emb
        
        self.decoder_layers = decoder_layers
        self.draft_layer_indexes = draft_layer_indexes
        self.tokenizer = tokenizer        
        self.config = config
    
    def forward(self, layer_idx, input_ids: torch.LongTensor, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None,
                fractal_key_value: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
                output_attentions: Optional[bool] = False,
                use_cache: Optional[bool] = False, cache_position: Optional[torch.LongTensor] = None,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, draft_len=0,
                prefix_len_list: Optional[List[int]] = None,):
        
        if cache_position is None:
                    past_seen_tokens = fractal_key_value.get_seq_length() if fractal_key_value is not None else 0
                    cache_position = torch.arange(
                        past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
                    )
                    

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
            
        if use_cache:
            if fractal_key_value is None:
                fractal_key_value = FractalCache()
                
        causal_mask = self._update_causal_mask(
            attention_mask, hidden_states, cache_position, fractal_key_value, output_attentions)  
        
        if causal_mask is not None:
            print("causal_mask.shape", causal_mask.shape)
        
        for i in range(layer_idx, len(self.decoder_layers)):
            hidden_states = self.decoder_layers[i].forward(
                hidden_states=hidden_states,
                attention_mask=causal_mask if use_cache else attention_mask,
                position_ids=position_ids if position_ids is not None else None,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                use_cache=use_cache,
                past_key_value=fractal_key_value,
                cache_position=cache_position,
            )[0]
        
        last_layer_out = hidden_states.to(self.lm_head_device)
        normed = self.fractal_norm(last_layer_out)
        
        # fractal_logits = self.lm_head(normed)
        fractal_logits = self.lm_head(
            self.fractal_norm(last_layer_out).to(self.lm_head.weight.dtype)
        )

        
        if use_cache:
            cache_batch_split = None
            if fractal_key_value.get_seq_length() > 0:
                cache_batch_split = fractal_key_value.batch_split(len(prefix_len_list), 1)
        
        for batch_idx in range(len(prefix_len_list)):
            prefix_len = prefix_len_list[batch_idx]
            
            if use_cache and cache_batch_split is not None:
                if cache_batch_split[batch_idx].get_seq_length() > 0:
                    prefix_len = 1
                fractal_key_value = FractalCache.from_batch_splits(cache_batch_split)
            # start = prefix_len - 1
            # stop  = prefix_len + draft_len - 1
            # logits_slice = fractal_logits[batch_idx, start:stop]
            # pred = logits_slice.argmax(dim=-1)
            # tgt = pred.unsqueeze(0)
            # need_len = prefix_len + draft_len
            # cur_len  = input_ids.shape[1]
            # if need_len > cur_len:                       # 시퀀스 끝에 새로 붙이기
            #     pad = tgt[:, cur_len-prefix_len : ]
            #     input_ids = torch.cat([input_ids, pad], dim=1)
            # input_ids[batch_idx, prefix_len : prefix_len+draft_len] = pred    
            for i in range(prefix_len, prefix_len + draft_len):
                token_logits = fractal_logits[batch_idx, i-1, :]
                # token_prob = torch.softmax(token_logits, dim=-1)
                token_prob = token_logits
                token_pred = torch.argmax(token_prob).item()
                
                if i >= input_ids.shape[1]:
                    token_tensor = torch.full(
                    (input_ids.shape[0], 1),
                    token_pred,
                    device=input_ids.device,
                    dtype=input_ids.dtype # input_ids와 동일한 데이터 타입 사용
                    )
                    input_ids = torch.cat([input_ids, token_tensor], dim=1)

                else:
                    input_ids[batch_idx, i] = token_pred
        
        if use_cache:
            fractal_key_value.crop(-draft_len)   
        
        new_seq_len = input_ids.shape[1]
        fractal_emb = self.fractal_embedding(input_ids)
        hidden_states = fractal_emb
        position_ids = torch.arange(
            new_seq_len, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        if cache_position is None:
            past_seen_tokens = fractal_key_value.get_seq_length() if fractal_key_value is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )
        
        causal_mask = self._update_causal_mask(
            attention_mask, hidden_states, cache_position, fractal_key_value, output_attentions)
        
        for i in range(layer_idx):
            layer_out = self.decoder_layers[i].forward(
                hidden_states=hidden_states,
                attention_mask=causal_mask if use_cache else attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                past_key_value=fractal_key_value,
            )
            
            if output_attentions:
                hidden_states, attn_weight = layer_out
            else:
                hidden_states = layer_out[0]

        return hidden_states, attention_mask, position_embeddings, position_ids, input_ids
    
    
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        
        else :
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length
            )
            

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask
    

class FractalLlamaDecoderLayer(torch.nn.Module):

    def __init__(
        self,
        config,
        experiement_config,
        layer_idx,               # 현재 레이어 인덱스
        decoder_layer: LlamaDecoderLayer,               # 기존 레이어 객체
        fractal_module: FractalModule,  # Fractal 모듈
    ):
        super().__init__()
        self.config=config
        self.experiement_config = experiement_config
        self.decoder_layer = decoder_layer
        self.layer_idx = layer_idx
        
        self.fractal_module = fractal_module
        self.draft_len = experiement_config.draft_len

        self.is_verify = False
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        hidden_states: torch.Tensor,      
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[object] = None,
        fractal_key_value: Optional[object] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        prefix_len_list: Optional[List[int]] = None,
        fidx: Optional[int] = None,
        skip_original: Optional[bool] = False,
        draft_len_current: int = 0,
        **kwargs,
    ):
        if prefix_len_list is not None:
            prefix_len = prefix_len_list[0]
            
        if not self.is_verify:
            fractal_out = self.fractal_module(
            layer_idx=self.layer_idx,
            input_ids=input_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask if attention_mask is not None else None,
            position_ids=position_ids if position_ids is not None else None,
            position_embeddings=(
                tuple(pos_emb for pos_emb in position_embeddings)
                if position_embeddings is not None else None
            ),
            fractal_key_value=fractal_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position if cache_position is not None else None,
            draft_len=draft_len_current + fidx+1,
            prefix_len_list=prefix_len_list,
            **kwargs,
            )
            prefix_len = prefix_len_list[0]
            if skip_original: # cache 구현용, 의미없음
                return None
            
            fractal_hidden_states, attention_mask, position_embeddings, position_ids, input_ids = fractal_out
            
            if hidden_states.device != fractal_hidden_states.device:
                fractal_hidden_states = fractal_hidden_states.to(hidden_states.device)
        
            new_hiddens=[]
            for batch_idx in range(len(prefix_len_list)):
                prefix_len = prefix_len_list[batch_idx]
                
                original_part = hidden_states[batch_idx, :, :]  
                prefix_part   = original_part[:(prefix_len), :]     

                fractal_part = fractal_hidden_states[batch_idx, (prefix_len) : (prefix_len) + (self.draft_len + fidx+1), :]
                
                rest_part = original_part[(prefix_len) + (self.draft_len + fidx+1) :, :]
                
                combined_part = torch.cat((prefix_part, fractal_part, rest_part), dim=0)
                combined_part = combined_part.unsqueeze(0)  # shape=(1, new_seq_len, hidden_dim)

                new_hiddens.append(combined_part)
                
            hidden_states = torch.cat(new_hiddens, dim=0)

        results = self.decoder_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask if attention_mask is not None else None,
            position_ids=position_ids if position_ids is not None else None,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position if cache_position is not None else None,
            position_embeddings=(
                tuple(pos_emb for pos_emb in position_embeddings)
                if position_embeddings is not None else None
            ),
            **kwargs,
        )

        if output_attentions:
            normal_out, self_attn_weights = results
        else:
            normal_out = results[0]
            self_attn_weights = None

        return (normal_out, self_attn_weights, position_embeddings, position_ids, input_ids)  if output_attentions else (normal_out, position_embeddings, position_ids, input_ids)
    

def load_quantized_model(
    model_name: str,
    cache_dir: str,
    device_map: str,
    experiment_config,
):
    bnb_cfg = None
    if experiment_config.decomp_method == 'quant_4bit':
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, # 4bit 할 것이냐
            bnb_4bit_compute_dtype=torch.float16, #bfloat16 or float16
            bnb_4bit_quant_type="nf4", # nf4 or fp4
            bnb_4bit_use_double_quant=False, # nf4 + fp4 이중 양자화 사용여부(성능은 크게 안다르고, 양자화 정확도 때문인듯)
        )
        
    elif experiment_config.decomp_method == 'quant_8bit':
        bnb_cfg = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_quant_type="nf4", # n4가 표준
            bnb_8bit_use_double_quant=True,
        )
    else:
        raise ValueError("Invalid decomp_method. Choose either 'quant_4bit' or 'quant_8bit'.")
    
    quant_model = LlamaForCausalLM.from_pretrained(
        model_name,
        cache_dir   = cache_dir,
        device_map  = device_map,
        quantization_config = bnb_cfg,
    )
    
    return quant_model

def load_model_with_fractal(
    model_name: str,
    cache_dir: str,
    device_map: str,
    tokenizer,
    experiment_config,
):
    print(f"[INFO] Loading model: {model_name}")
    base_model = FractalLlamaForCausalLM.from_pretrained(
        model_name, cache_dir=cache_dir, device_map=device_map
    )
    
    base_model.resize_token_embeddings(len(tokenizer))
    print(f"[INFO] Model loaded with device_map={device_map}")
    
    base_model.config.use_cache = experiment_config.use_cache
    print("use_cache: ", base_model.config.use_cache)
    
    total_layers = len(base_model.model.layers)
    norm_layer = base_model.model.norm
    rotary_emb = base_model.model.rotary_emb
    
    quant_model = load_quantized_model(
        model_name=model_name,
        cache_dir=cache_dir,
        device_map=device_map,
        experiment_config=experiment_config
    )
    quant_embedding_layer = quant_model.get_input_embeddings()
    quant_lm_head = quant_model.get_output_embeddings()
    
    decomposed_layers = quant_model.model.layers
    base_model.model.set_quantized_layers(decomposed_layers)
            
    print(f"[INFO] Total numbers including decomposed layers: {len(decomposed_layers)}")
    
    fractal_module = FractalModule(
        config=base_model.config,
        embedding_layer=quant_embedding_layer,
        shared_lm_head=quant_lm_head,
        decoder_layers=decomposed_layers,
        norm_layer=norm_layer,
        tokenizer=tokenizer,
        rotary_emb=rotary_emb,
    )
    fractal_layer_indexes = experiment_config.draft_layer_indexes
    if len(fractal_layer_indexes) == 0:
        step = max(total_layers // experiment_config.num_draft_layers, 1)
        fractal_layer_indexes = list(range(0, total_layers, step))[:experiment_config.num_draft_layers]

    print(f"[INFO] Replacing layers {fractal_layer_indexes} with FractalLlamaDecoderLayer")

    for idx in fractal_layer_indexes:
        device = next(base_model.model.layers[idx].parameters()).device
        decoder_layer = base_model.model.layers[idx]

        replacement_layer = FractalLlamaDecoderLayer(
            config=base_model.config,
            experiement_config=experiment_config,
            layer_idx=idx,
            decoder_layer=decoder_layer,
            fractal_module=fractal_module,
        )
        
        base_model.model.layers[idx] = replacement_layer
        print(f"[DEBUG] Replaced layer {idx} -> FractalLlamaDecoderLayer on device {device}")
            
    return base_model

def set_verify_mode(model, is_verify: bool):
    if isinstance(model, FractalLlamaForCausalLM):
        for layer in model.model.layers:
            if isinstance(layer, FractalLlamaDecoderLayer):
                layer.is_verify = is_verify
    elif isinstance(model, FractalLlamaModel):
        for layer in model.layers:
            if isinstance(layer, FractalLlamaDecoderLayer):
                layer.is_verify = is_verify
    else:
        raise ValueError("Invalid model type for Fractal mode setting.")