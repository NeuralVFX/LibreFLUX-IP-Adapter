from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
import torch.nn.functional as F
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention_processor import Attention
import inspect
from functools import partial
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
 
 
def apply_rope(xq, xk, freqs_cis):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
 

class IPFluxAttnProcessor2_0(nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""
 
    def __init__(self,hidden_size,cross_attention_dim, scale=1.0, num_tokens=8,num_heads=12):
        super().__init__()
 
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
 
        self.hidden_size = hidden_size
        self.scale = scale
        self.num_tokens = num_tokens
        self.num_heads = num_heads
 
        self.to_k_ip = nn.Linear(cross_attention_dim, cross_attention_dim, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, cross_attention_dim, bias=False)
 
        self.add_k_proj_ip = nn.Linear(cross_attention_dim, cross_attention_dim, bias=False)
        self.add_v_proj_ip = nn.Linear(cross_attention_dim, cross_attention_dim, bias=False)
 
        self.proj_ip = nn.Linear(cross_attention_dim, cross_attention_dim, bias=False)
        self.proj_encoder_hidden_states_ip = nn.Linear(cross_attention_dim, cross_attention_dim, bias=False)
        self.proj_encoder_passthrough_ip = nn.Linear(cross_attention_dim, cross_attention_dim, bias=False)
        #zero init proj
        nn.init.zeros_(self.proj_ip.weight)
        nn.init.zeros_(self.proj_encoder_hidden_states_ip.weight)
        nn.init.eye_(self.proj_encoder_passthrough_ip.weight)
 
 
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        ip_encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        layer_scale: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
 
        batch_size = encoder_hidden_states.shape[0]
 
        original_hidden_states = hidden_states
        ip_encoder_hidden_states_original = ip_encoder_hidden_states
        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
 
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
 
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
 
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
 
        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
 
        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
 
        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
 
        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
 
        if image_rotary_emb is not None:
            # YiYi to-do: update uising apply_rotary_emb
            # from ..embeddings import apply_rotary_emb
            # query = apply_rotary_emb(query, image_rotary_emb)
            # key = apply_rotary_emb(key, image_rotary_emb)
            query, key = apply_rope(query, key, image_rotary_emb)
 
 
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
 
        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )
 
        ip_key = self.to_k_ip(original_hidden_states)
        ip_value = self.to_v_ip(original_hidden_states)
 
        ip_encoder_hidden_states_key_proj = self.add_k_proj_ip(ip_encoder_hidden_states)
        ip_encoder_hidden_states_value_proj = self.add_v_proj_ip(ip_encoder_hidden_states)
 
 
 
        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
 
 
        ip_encoder_hidden_states_key_proj = ip_encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        ip_encoder_hidden_states_value_proj = ip_encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
 
        #if attn.norm_added_k is not None:
        #    ip_encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
 
 
        ip_key = torch.cat([ip_encoder_hidden_states_key_proj, ip_key], dim=2)
        ip_value = torch.cat([ip_encoder_hidden_states_value_proj, ip_value], dim=2)
 
 
 
        ip_hidden_states = F.scaled_dot_product_attention(query, ip_key, ip_value, dropout_p=0.0, is_causal=False)
        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(query.dtype)
 
        ip_encoder_hidden_states, ip_hidden_states = (
            ip_hidden_states[:, : encoder_hidden_states.shape[1]],
            ip_hidden_states[:, encoder_hidden_states.shape[1] :],
        )
 
        ip_hidden_states = self.proj_ip(ip_hidden_states)
        ip_encoder_hidden_states = self.proj_encoder_hidden_states_ip(ip_encoder_hidden_states)
        ip_encoder_hidden_states_original = self.proj_encoder_passthrough_ip(ip_encoder_hidden_states_original)
 
        layer_scale = layer_scale.view(-1, 1, 1)
        hidden_states = hidden_states + (self.scale * ip_hidden_states * layer_scale)
        encoder_hidden_states = encoder_hidden_states + (self.scale * ip_encoder_hidden_states * layer_scale)
 
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
 
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
 
        return hidden_states, encoder_hidden_states, ip_encoder_hidden_states_original
 
 
 