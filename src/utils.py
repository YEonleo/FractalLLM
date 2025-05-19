import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import evaluate  # pip install evaluate
# Evaluate metrics

from typing import Optional, List, Tuple, Dict, Any
from transformers.cache_utils import DynamicCache, Cache

from transformers.models.llama.modeling_llama import LlamaAttention


bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")


def gumbel_softmax(logits, tau=1.0, hard=False):
    device = logits.device
    gumbels = -torch.empty_like(logits, device=device).exponential_().log()
    y = logits + gumbels
    y = F.softmax(y / tau, dim=-1)

    if hard:
        y_hard = torch.zeros_like(y, device=device).scatter_(
            -1, y.argmax(dim=-1, keepdim=True), 1.0
        )
        y = (y_hard - y).detach() + y

    return y

class FlopsCounter:
    def __init__(self, model):
        self.model = model
        self.total_flops = 0
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Hook the linear layers
                handle = module.register_forward_hook(self._linear_flops_hook)
                self.handles.append(handle)
            elif isinstance(module, LlamaAttention):
                # Hook the custom LLaMA attention
                handle = module.register_forward_hook(self._llama_attn_flops_hook)
                self.handles.append(handle)

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def reset(self):
        self.total_flops = 0

    def get_total_flops(self):
        return self.total_flops

    def _linear_flops_hook(self, module, input, output):
        
        x = input[0]
        if x.dim() == 3:
            batch_size, seq_len, in_features = x.shape
        elif x.dim() == 2:
            # no seq dimension (possibly intermediate usage),
            # treat seq_len=1 for flop count
            batch_size, in_features = x.shape
            seq_len = 1
        else:
            # fallback
            batch_size = x.shape[0]
            seq_len = 1
            in_features = module.in_features
        
        out_features = module.out_features
        # Multiply-add pairs => *2. Adjust if you prefer otherwise
        flops = 2.0 * batch_size * seq_len * in_features * out_features
        self.total_flops += flops

    def _llama_attn_flops_hook(self, module, input, output):

        attn_output = output[0]  # shape [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = attn_output.shape
        if hasattr(module, "num_heads"):
            num_heads = module.num_heads
        elif hasattr(module, "config") and hasattr(module.config, "num_attention_heads"):
            num_heads = module.config.num_attention_heads
        else:
            raise ValueError("Cannot find `num_heads` or `num_attention_heads` in LlamaAttention.")

        head_dim = hidden_dim // num_heads
        
        # QK^T cost:
        qk_flops = 2.0 * batch_size * num_heads * seq_len * seq_len * head_dim
        # attn_weights @ V cost:
        av_flops = 2.0 * batch_size * num_heads * seq_len * seq_len * head_dim
        self.total_flops += (qk_flops + av_flops)


def measure_generation_time(model, input_ids, max_length=50):
    
    start = time.time()
    with torch.inference_mode():
        out = model.generate(input_ids, max_new_tokens=max_length)
    end = time.time()

    total_time = end - start
    new_tokens = out.shape[-1] - input_ids.shape[-1]  
    
    if new_tokens == 0:
        time_per_new_token = 0.0
    else:
        time_per_new_token = total_time / new_tokens
    
    return out, total_time, time_per_new_token

def compute_metrics(prediction: str, reference: str):
    bleu_results = bleu_metric.compute(predictions=[prediction], references=[[reference]])
    bleu_score = bleu_results["bleu"]
    rouge_results = rouge_metric.compute(predictions=[prediction], references=[reference])
    rouge1_score = rouge_results["rouge1"]
    rouge2_score = rouge_results["rouge2"]
    rougeL_score = rouge_results["rougeL"]
    return bleu_score, rouge1_score, rouge2_score, rougeL_score


def compute_causal_mask(
    attention_mask: Optional[torch.Tensor],
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    output_attentions: bool,
    config,
    training: bool,
) -> Optional[torch.Tensor]:
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter
    from transformers.cache_utils import StaticCache

    if config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and (attention_mask == 0.0).any():
            return attention_mask
        return None

    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    using_static_cache = isinstance(past_key_values, StaticCache)

    if config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
        if AttentionMaskConverter._ignore_causal_mask_sdpa(
            attention_mask,
            inputs_embeds=input_tensor,
            past_key_values_length=past_seen_tokens,
            is_training=training,
        ):
            return None

    dtype, device = input_tensor.dtype, input_tensor.device
    sequence_length = input_tensor.shape[1]
    target_length = (
        attention_mask.shape[-1]
        if attention_mask is not None
        else past_seen_tokens + sequence_length + 1
    )

    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full(
        (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
    )
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)

    if attention_mask is not None:
        causal_mask = causal_mask.clone()
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
            padding_mask, min_dtype
        )

    if (
        config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type == "cuda"
        and not output_attentions
    ):
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

    return causal_mask



class FractalCache(DynamicCache):
    def crop_by_index(self, layer_idx: int, max_length: int):

        if not (0 <= layer_idx < len(self.key_cache)):
            raise IndexError(f"Invalid layer_idx {layer_idx}. Cache has {len(self.key_cache)} layers.")

        key_cache_layer = self.key_cache[layer_idx]
        value_cache_layer = self.value_cache[layer_idx]

        is_valid_tensor_layer = isinstance(key_cache_layer, torch.Tensor) and isinstance(value_cache_layer, torch.Tensor)

        if not is_valid_tensor_layer:
            return # 자를 대상이 없으므로 종료

        current_layer_len = key_cache_layer.shape[-2]
        if max_length < 0:
            max_length = current_layer_len + max_length # 음수값 더하기 (빼기)
            if max_length < 0: # 결과가 음수가 되지 않도록 0으로 설정
                max_length = 0

        if current_layer_len <= max_length:
            return # 자를 필요 없으므로 종료

        self.key_cache[layer_idx] = key_cache_layer[..., :max_length, :]
        self.value_cache[layer_idx] = value_cache_layer[..., :max_length, :]