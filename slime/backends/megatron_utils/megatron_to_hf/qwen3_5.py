import re

import torch


def convert_qwen3_5_to_hf(args, name, param):
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.language_model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.language_model.norm.weight", param)]

    head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    match = re.match(r"module\.module\.decoder\.layers\.(\d+)\.(.+)", name)
    if match:
        layer_idx, rest = match.groups()

        if rest == "self_attention.linear_proj.weight":
            return [(f"model.language_model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        if rest == "self_attention.linear_qkv.weight":
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(
                param, split_size_or_sections=[2 * value_num_per_group, 1, 1], dim=1
            )
            q_param = (
                q_param.reshape(args.num_query_groups, 2, value_num_per_group, head_dim, args.hidden_size)
                .transpose(1, 2)
                .reshape(-1, args.hidden_size)
            )
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"model.language_model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.language_model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.language_model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]
        if rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.language_model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.language_model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        if rest == "mlp.linear_fc2.weight":
            return [(f"model.language_model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        if rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.language_model.layers.{layer_idx}.input_layernorm.weight", param)]
        if rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"model.language_model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        if rest == "pre_mlp_layernorm.weight":
            return [(f"model.language_model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        if rest == "self_attention.q_layernorm.weight":
            return [(f"model.language_model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        if rest == "self_attention.k_layernorm.weight":
            return [(f"model.language_model.layers.{layer_idx}.self_attn.k_norm.weight", param)]

        if rest.startswith("self_attention."):
            sub = rest[len("self_attention.") :]
            if sub in {
                "input_layernorm.weight",
                "linear_attn.A_log",
                "linear_attn.conv1d.weight",
                "linear_attn.dt_bias",
                "linear_attn.in_proj_a.weight",
                "linear_attn.in_proj_b.weight",
                "linear_attn.in_proj_qkv.weight",
                "linear_attn.in_proj_z.weight",
                "linear_attn.norm.weight",
                "linear_attn.out_proj.weight",
            }:
                return [(f"model.language_model.layers.{layer_idx}.{sub}", param)]

    raise ValueError(f"Unknown Qwen3.5 parameter name: {name}")
