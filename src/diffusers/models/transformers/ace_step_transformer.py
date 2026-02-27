# Copyright 2025 The ACE-Step Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ..attention import AttentionMixin
from ..attention_dispatch import dispatch_attention_fn
from ..embeddings import TimestepEmbedding, Timesteps, apply_rotary_emb, get_1d_rotary_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _create_4d_mask(
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    is_sliding_window: bool = False,
    is_causal: bool = True,
) -> torch.Tensor:
    """
    Create a 4D attention mask compatible with SDPA and eager attention.

    Supports causal/bidirectional attention with optional sliding window.

    Returns:
        Tensor of shape `[batch, 1, seq_len, seq_len]` with `0.0` for visible positions and `-inf` for masked ones.
    """
    indices = torch.arange(seq_len, device=device)
    diff = indices.unsqueeze(1) - indices.unsqueeze(0)
    valid_mask = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool)

    if is_causal:
        valid_mask = valid_mask & (diff >= 0)

    if is_sliding_window and sliding_window is not None:
        if is_causal:
            valid_mask = valid_mask & (diff <= sliding_window)
        else:
            valid_mask = valid_mask & (torch.abs(diff) <= sliding_window)

    valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)

    if attention_mask is not None:
        padding_mask_4d = attention_mask.view(attention_mask.shape[0], 1, 1, seq_len).to(torch.bool)
        valid_mask = valid_mask & padding_mask_4d

    min_dtype = torch.finfo(dtype).min
    mask_tensor = torch.full(valid_mask.shape, min_dtype, dtype=dtype, device=device)
    mask_tensor.masked_fill_(valid_mask, 0.0)
    return mask_tensor


class AceStepMLP(nn.Module):
    """MLP (SwiGLU) used in ACE-Step transformer layers."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class AceStepAttnProcessor:
    """
    Attention processor for ACE-Step models.

    Handles self-attention and cross-attention with RoPE, grouped query attention,
    and dispatches to the appropriate attention backend (Flash Attention, SDPA, etc.).
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0 or higher.")

    def __call__(
        self,
        attn: "AceStepAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        is_cross_attention = encoder_hidden_states is not None

        query = attn.to_q(hidden_states)
        kv_input = encoder_hidden_states if is_cross_attention else hidden_states
        key = attn.to_k(kv_input)
        value = attn.to_v(kv_input)

        # Reshape to multi-head: [B, S, H*D] -> [B, S, H, D]
        query = query.unflatten(-1, (attn.heads, attn.head_dim))
        key = key.unflatten(-1, (attn.kv_heads, attn.head_dim))
        value = value.unflatten(-1, (attn.kv_heads, attn.head_dim))

        # QK normalization
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Apply RoPE (only for self-attention)
        if not is_cross_attention and position_embeddings is not None:
            query = apply_rotary_emb(
                query, position_embeddings, use_real=True, use_real_unbind_dim=-2, sequence_dim=1
            )
            key = apply_rotary_emb(
                key, position_embeddings, use_real=True, use_real_unbind_dim=-2, sequence_dim=1
            )

        # GQA: repeat KV heads
        if attn.heads != attn.kv_heads:
            key = key.repeat_interleave(attn.heads // attn.kv_heads, dim=2)
            value = value.repeat_interleave(attn.heads // attn.kv_heads, dim=2)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=attn.dropout if attn.training else 0.0,
            scale=attn.scale,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        # Reshape back: [B, S, H, D] -> [B, S, H*D]
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class AceStepAttention(nn.Module):
    """
    Multi-headed attention module for the ACE-Step model.

    Uses the AttnProcessor pattern for backend flexibility (Flash Attention, SDPA, etc.)
    and supports self-attention and cross-attention with RMSNorm on query/key.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        processor: Optional[AceStepAttnProcessor] = None,
    ):
        super().__init__()
        self.heads = num_attention_heads
        self.kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.dropout = attention_dropout

        inner_dim = num_attention_heads * head_dim
        inner_kv_dim = num_key_value_heads * head_dim

        self.to_q = nn.Linear(hidden_size, inner_dim, bias=attention_bias)
        self.to_k = nn.Linear(hidden_size, inner_kv_dim, bias=attention_bias)
        self.to_v = nn.Linear(hidden_size, inner_kv_dim, bias=attention_bias)
        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, hidden_size, bias=attention_bias), nn.Dropout(attention_dropout)]
        )

        self.norm_q = RMSNorm(head_dim, eps=rms_norm_eps)
        self.norm_k = RMSNorm(head_dim, eps=rms_norm_eps)

        self.processor = processor or AceStepAttnProcessor()

    def get_processor(self) -> AceStepAttnProcessor:
        return self.processor

    def set_processor(self, processor: AceStepAttnProcessor):
        self.processor = processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )


class AceStepTransformerBlock(nn.Module):
    """
    DiT (Diffusion Transformer) block for the ACE-Step model.

    Implements a transformer block with:
    1. Self-attention with adaptive layer norm (AdaLN)
    2. Cross-attention for conditioning on encoder outputs
    3. Feed-forward MLP with adaptive layer norm

    Uses scale-shift modulation from timestep embeddings for adaptive normalization.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        sliding_window: Optional[int] = None,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        # Self-attention
        self.self_attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = AceStepAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            rms_norm_eps=rms_norm_eps,
        )

        # Cross-attention
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.cross_attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.cross_attn = AceStepAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
                rms_norm_eps=rms_norm_eps,
            )

        # Feed-forward MLP
        self.mlp_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = AceStepMLP(hidden_size, intermediate_size)

        # Scale-shift table for adaptive layer norm (6 values)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, hidden_size) / hidden_size**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (self.scale_shift_table + temb).chunk(
            6, dim=1
        )

        # Self-attention with AdaLN
        norm_hidden_states = (self.self_attn_norm(hidden_states) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.self_attn(
            hidden_states=norm_hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = (hidden_states + attn_output * gate_msa).type_as(hidden_states)

        # Cross-attention
        if self.use_cross_attention and encoder_hidden_states is not None:
            norm_hidden_states = self.cross_attn_norm(hidden_states).type_as(hidden_states)
            attn_output = self.cross_attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = hidden_states + attn_output

        # Feed-forward MLP with AdaLN
        norm_hidden_states = (self.mlp_norm(hidden_states) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
        ff_output = self.mlp(norm_hidden_states)
        hidden_states = (hidden_states + ff_output * c_gate_msa).type_as(hidden_states)

        return hidden_states


class AceStepTransformer1DModel(ModelMixin, ConfigMixin, AttentionMixin):
    """
    The Diffusion Transformer (DiT) model for ACE-Step 1.5 music generation.

    This model generates audio latents conditioned on text, lyrics, and timbre. It uses patch-based processing with
    transformer layers, timestep conditioning via AdaLN, and cross-attention to encoder outputs.

    Parameters:
        hidden_size (`int`, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, defaults to 6144):
            Dimension of the MLP intermediate representations.
        num_hidden_layers (`int`, defaults to 24):
            Number of DiT transformer layers.
        num_attention_heads (`int`, defaults to 16):
            Number of attention heads for query states.
        num_key_value_heads (`int`, defaults to 8):
            Number of attention heads for key and value states (for grouped query attention).
        head_dim (`int`, defaults to 128):
            Dimension of each attention head.
        in_channels (`int`, defaults to 192):
            Number of input channels (context_latents + hidden_states concatenated).
        audio_acoustic_hidden_dim (`int`, defaults to 64):
            Output dimension of the model (acoustic latent dimension).
        patch_size (`int`, defaults to 2):
            Patch size for input patchification.
        rope_theta (`float`, defaults to 1000000.0):
            Base period of the RoPE embeddings.
        attention_bias (`bool`, defaults to `False`):
            Whether to use bias in attention projection layers.
        attention_dropout (`float`, defaults to 0.0):
            Dropout probability for attention weights.
        rms_norm_eps (`float`, defaults to 1e-6):
            Epsilon for RMS normalization.
        sliding_window (`int`, defaults to 128):
            Sliding window size for local attention layers.
        layer_types (`List[str]`, *optional*):
            Attention pattern for each layer. Defaults to alternating `"sliding_attention"` and `"full_attention"`.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["AceStepTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        in_channels: int = 192,
        audio_acoustic_hidden_dim: int = 64,
        patch_size: int = 2,
        rope_theta: float = 1000000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        sliding_window: int = 128,
        layer_types: Optional[List[str]] = None,
    ):
        super().__init__()
        self.patch_size = patch_size

        # Determine layer types
        if layer_types is None:
            layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(num_hidden_layers)
            ]

        # DiT transformer layers
        self.layers = nn.ModuleList(
            [
                AceStepTransformerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    attention_bias=attention_bias,
                    attention_dropout=attention_dropout,
                    rms_norm_eps=rms_norm_eps,
                    sliding_window=sliding_window if layer_types[i] == "sliding_attention" else None,
                    use_cross_attention=True,
                )
                for i in range(num_hidden_layers)
            ]
        )

        # Store layer types for mask selection
        self._layer_types = layer_types

        # Input projection (patchify)
        self.proj_in_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        # Timestep embeddings: sinusoidal -> MLP projection
        self.timestep_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.time_embed = TimestepEmbedding(in_channels=256, time_embed_dim=hidden_size, act_fn="silu")
        self.adaln_single = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))

        self.timestep_proj_r = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.time_embed_r = TimestepEmbedding(in_channels=256, time_embed_dim=hidden_size, act_fn="silu")
        self.adaln_single_r = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))

        # Condition projection
        self.condition_embedder = nn.Linear(hidden_size, hidden_size, bias=True)

        # Output (de-patchify)
        self.norm_out = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.proj_out_conv = nn.ConvTranspose1d(
            in_channels=hidden_size,
            out_channels=audio_acoustic_hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, hidden_size) / hidden_size**0.5)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_r: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        context_latents: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`AceStepTransformer1DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, channels)`):
                Noisy latent input for the diffusion process.
            timestep (`torch.Tensor` of shape `(batch_size,)`):
                Current diffusion timestep `t`.
            timestep_r (`torch.Tensor` of shape `(batch_size,)`):
                Reference timestep `r` (set equal to `t` for standard inference).
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, encoder_seq_len, hidden_size)`):
                Conditioning embeddings from the condition encoder (text + lyrics + timbre).
            context_latents (`torch.Tensor` of shape `(batch_size, seq_len, context_dim)`):
                Context latents (source latents concatenated with chunk masks).
            attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the hidden states sequence.
            encoder_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the encoder hidden states.
            return_dict (`bool`, defaults to `True`):
                Whether to return a `Transformer2DModelOutput` or a plain tuple.

        Returns:
            `Transformer2DModelOutput` or `tuple`: The predicted velocity field for flow matching.
        """
        # Compute timestep embeddings
        t_sinusoidal = self.timestep_proj(timestep).to(dtype=hidden_states.dtype)
        temb_t = self.time_embed(t_sinusoidal)
        timestep_proj_t = self.adaln_single(temb_t).unflatten(1, (6, -1))

        r_sinusoidal = self.timestep_proj_r(timestep - timestep_r).to(dtype=hidden_states.dtype)
        temb_r = self.time_embed_r(r_sinusoidal)
        timestep_proj_r = self.adaln_single_r(temb_r).unflatten(1, (6, -1))

        temb = temb_t + temb_r
        timestep_proj = timestep_proj_t + timestep_proj_r

        # Concatenate context latents with hidden states
        hidden_states = torch.cat([context_latents, hidden_states], dim=-1)
        original_seq_len = hidden_states.shape[1]

        # Pad if sequence length is not divisible by patch_size
        pad_length = 0
        if hidden_states.shape[1] % self.patch_size != 0:
            pad_length = self.patch_size - (hidden_states.shape[1] % self.patch_size)
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_length), mode="constant", value=0)

        # Patchify: [B, T, C] -> [B, C, T] -> conv -> [B, C', T'] -> [B, T', C']
        hidden_states = self.proj_in_conv(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Project encoder hidden states
        encoder_hidden_states = self.condition_embedder(encoder_hidden_states)

        # Position embeddings using get_1d_rotary_pos_embed
        seq_len = hidden_states.shape[1]
        position_embeddings = get_1d_rotary_pos_embed(
            self.config.head_dim, seq_len, theta=self.config.rope_theta, use_real=True, repeat_interleave_real=False
        )

        # Build attention masks
        dtype = hidden_states.dtype
        device = hidden_states.device
        encoder_seq_len = encoder_hidden_states.shape[1]

        full_attn_mask = _create_4d_mask(
            seq_len=seq_len, dtype=dtype, device=device, attention_mask=None, is_causal=False
        )
        encoder_4d_mask = _create_4d_mask(
            seq_len=max(seq_len, encoder_seq_len), dtype=dtype, device=device, attention_mask=None, is_causal=False
        )
        encoder_4d_mask = encoder_4d_mask[:, :, :seq_len, :encoder_seq_len]

        sliding_attn_mask = _create_4d_mask(
            seq_len=seq_len,
            dtype=dtype,
            device=device,
            attention_mask=None,
            sliding_window=self.config.sliding_window,
            is_sliding_window=True,
            is_causal=False,
        )

        # Process through transformer layers
        for i, layer_module in enumerate(self.layers):
            layer_type = self._layer_types[i]
            if layer_type == "sliding_attention":
                layer_attn_mask = sliding_attn_mask
            else:
                layer_attn_mask = full_attn_mask

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    position_embeddings,
                    timestep_proj,
                    layer_attn_mask,
                    encoder_hidden_states,
                    encoder_4d_mask,
                )
            else:
                hidden_states = layer_module(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    temb=timestep_proj,
                    attention_mask=layer_attn_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_4d_mask,
                )

        # Adaptive output normalization
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (self.norm_out(hidden_states) * (1 + scale) + shift).type_as(hidden_states)

        # De-patchify: [B, T', C'] -> [B, C', T'] -> deconv -> [B, C, T] -> [B, T, C]
        hidden_states = self.proj_out_conv(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Crop to original sequence length
        hidden_states = hidden_states[:, :original_seq_len, :]

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
