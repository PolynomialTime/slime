import copy
import json
import os
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import mpu, tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers.activations import ACT2FN

try:
    # Same training kernels used by the existing qwen3_next Megatron plugin.
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ImportError:  # pragma: no cover - fail with a clear message at runtime.
    FusedRMSNormGated = None
    ShortConvolution = None
    chunk_gated_delta_rule = None


class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        # Qwen3.5 stores RMSNorm gamma in zero-centered form.
        # Effective scale is 1 + weight, matching HF Qwen3_5RMSNorm.
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = F.rms_norm(hidden_states.float(), (hidden_states.shape[-1],), None, self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(hidden_states)


def _load_qwen35_text_config(hf_checkpoint: str) -> dict:
    config_path = os.path.join(hf_checkpoint, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    text_config = config.get("text_config", config)
    rope_parameters = text_config.get("rope_parameters") or {}
    if "rope_scaling" not in text_config or text_config.get("rope_scaling") is None:
        text_config["rope_scaling"] = rope_parameters
    text_config.setdefault("rope_parameters", rope_parameters)
    return text_config


class Qwen3_5GatedDeltaNet(nn.Module):
    """Qwen3.5 Gated DeltaNet for Megatron varlen training.

    The public Qwen3.5 HF checkpoint stores split projections:
    in_proj_qkv / in_proj_z / in_proj_b / in_proj_a.  This differs from
    qwen3_next's fused in_proj_qkvz / in_proj_ba layout, so we keep a separate
    module instead of reusing the qwen3_next plugin.
    """

    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        if ShortConvolution is None or FusedRMSNormGated is None or chunk_gated_delta_rule is None:
            raise ImportError("Qwen3.5 linear attention requires fla modules in the slime image.")

        self.hidden_size = config["hidden_size"]
        self.num_v_heads = config["linear_num_value_heads"]
        self.num_k_heads = config["linear_num_key_heads"]
        self.head_k_dim = config["linear_key_head_dim"]
        self.head_v_dim = config["linear_value_head_dim"]
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config["linear_conv_kernel_dim"]
        self.layer_idx = layer_idx
        self.activation = config.get("hidden_act", "silu")
        self.act = ACT2FN[self.activation]
        self.layer_norm_epsilon = config.get("rms_norm_eps", 1e-6)

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ShortConvolution(
            hidden_size=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
        )

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = FusedRMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            activation=self.activation,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor = None) -> torch.Tensor:
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        mixed_qkv, _ = self.conv1d(x=mixed_qkv, cu_seqlens=cu_seqlens)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(query.shape[0], query.shape[1], self.num_k_heads, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], self.num_k_heads, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], self.num_v_heads, self.head_v_dim)
        z = z.reshape(z.shape[0], z.shape[1], self.num_v_heads, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            repeat = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        core_attn_out, _ = chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

        z_shape = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)
        return self.out_proj(core_attn_out)


class _VarlenLinearAttention(MegatronModule, ABC):
    def __init__(self, args, config, layer_number: int, cp_comm_type: str = "p2p", pg_collection=None):
        super().__init__(config=config)
        self.args = args
        self.config = config
        self.layer_number = layer_number
        self.hf_layer_idx = layer_number - 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert packed_seq_params is not None
        cu_seqlens = packed_seq_params.cu_seqlens_q

        if self.args.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, group=mpu.get_tensor_model_parallel_group()
            )

        if mpu.get_context_parallel_world_size() > 1:
            cp_size = mpu.get_context_parallel_world_size()
            hidden_states_list = dist.nn.all_gather(hidden_states, group=mpu.get_context_parallel_group())
            whole_hidden_states_list = []
            local_cu_seqlens = cu_seqlens // cp_size
            for i in range(len(cu_seqlens) - 1):
                seqlen = cu_seqlens[i + 1] - cu_seqlens[i]
                chunk_size = seqlen // 2 // cp_size
                whole_hidden_states_list.extend(
                    [hidden_states_list[cp_rank][local_cu_seqlens[i] : local_cu_seqlens[i] + chunk_size] for cp_rank in range(cp_size)]
                    + [
                        hidden_states_list[cp_rank][local_cu_seqlens[i] + chunk_size : local_cu_seqlens[i + 1]]
                        for cp_rank in range(cp_size)
                    ][::-1]
                )
            hidden_states = torch.cat(whole_hidden_states_list, dim=0)

        hidden_states = hidden_states.permute(1, 0, 2)
        output = self.hf_forward(hidden_states, packed_seq_params)
        output = output.permute(1, 0, 2)

        if mpu.get_context_parallel_world_size() > 1:
            cp_size = mpu.get_context_parallel_world_size()
            cp_rank = mpu.get_context_parallel_rank()
            output_list = []
            for i in range(len(cu_seqlens) - 1):
                seqlen = cu_seqlens[i + 1] - cu_seqlens[i]
                chunk_size = seqlen // 2 // cp_size
                seq = output[cu_seqlens[i] : cu_seqlens[i + 1]]
                chunks = torch.chunk(seq, 2 * cp_size, dim=0)
                output_list.append(chunks[cp_rank])
                output_list.append(chunks[2 * cp_size - 1 - cp_rank])
            output = torch.cat(output_list, dim=0)

        if self.args.sequence_parallel:
            output = tensor_parallel.scatter_to_sequence_parallel_region(
                output, group=mpu.get_tensor_model_parallel_group()
            )
        return output, None

    @abstractmethod
    def hf_forward(self, hidden_states, packed_seq_params):
        pass


class Attention(_VarlenLinearAttention):
    def __init__(self, args, config, layer_number: int, cp_comm_type: str = "p2p", pg_collection=None):
        super().__init__(args, config, layer_number, cp_comm_type, pg_collection)
        text_config = _load_qwen35_text_config(args.hf_checkpoint)
        self.linear_attn = Qwen3_5GatedDeltaNet(text_config, self.hf_layer_idx)
        self.input_layernorm = Qwen3_5RMSNorm(text_config["hidden_size"], eps=text_config.get("rms_norm_eps", 1e-6))

    def hf_forward(self, hidden_states, packed_seq_params):
        hidden_states = self.input_layernorm(hidden_states)
        return self.linear_attn(hidden_states=hidden_states, cu_seqlens=packed_seq_params.cu_seqlens_q)


def get_qwen3_5_spec(args, config, vp_stage):
    # Qwen3.5 alternates linear-attention and full-attention layers.  The
    # checkpoint writer must shard each layer under its real global layer id
    # instead of treating all decoder layers as homogeneous.
    config.hetereogenous_dist_checkpoint = True

    kwargs = {"use_transformer_engine": True}
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)

    assert config.pipeline_model_parallel_layout is None, "qwen3_5 plugin does not support pipeline layout yet"
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
    layer_types = _load_qwen35_text_config(args.hf_checkpoint)["layer_types"]

    for layer_id in range(num_layers_to_build):
        if layer_types[layer_id + offset] == "linear_attention":
            layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
            layer_specs.submodules.self_attention = ModuleSpec(module=Attention, params={"args": args})
            transformer_layer_spec.layer_specs[layer_id] = layer_specs
    return transformer_layer_spec
