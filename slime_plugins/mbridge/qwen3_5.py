import torch
from mbridge.core import register_model
from mbridge.models import Qwen2Bridge

# Registers qwen3_5 AutoConfig classes with transformers.AutoConfig in the
# slime image, where upstream transformers 4.57 does not know this model_type.
try:  # pragma: no cover - depends on the runtime image.
    import sglang.srt.utils.hf_transformers_utils  # noqa: F401
except Exception:
    pass


@register_model("qwen3_5")
class Qwen3_5Bridge(Qwen2Bridge):
    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.language_model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }
    _ATTENTION_MAPPING = (
        {
            "self_attention.linear_proj.weight": [
                "model.language_model.layers.{layer_number}.self_attn.o_proj.weight"
            ],
            "self_attention.linear_qkv.layer_norm_weight": [
                "model.language_model.layers.{layer_number}.input_layernorm.weight"
            ],
            "self_attention.q_layernorm.weight": [
                "model.language_model.layers.{layer_number}.self_attn.q_norm.weight"
            ],
            "self_attention.k_layernorm.weight": [
                "model.language_model.layers.{layer_number}.self_attn.k_norm.weight"
            ],
            "self_attention.linear_qkv.weight": [
                "model.language_model.layers.{layer_number}.self_attn.q_proj.weight",
                "model.language_model.layers.{layer_number}.self_attn.k_proj.weight",
                "model.language_model.layers.{layer_number}.self_attn.v_proj.weight",
            ],
        }
        | {
            f"self_attention.{weight_name}": [
                "model.language_model.layers.{layer_number}." + weight_name
            ]
            for weight_name in [
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
            ]
        }
    )
    _MLP_MAPPING = {
        "mlp.linear_fc1.weight": [
            "model.language_model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.language_model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "mlp.linear_fc1.layer_norm_weight": [
            "model.language_model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "mlp.linear_fc2.weight": ["model.language_model.layers.{layer_number}.mlp.down_proj.weight"],
    }

    @property
    def text_config(self):
        return getattr(self.hf_config, "text_config", self.hf_config)

    def _get_hf_shared_weight_keys(self):
        return []

    def _build_config(self):
        return self._build_base_config(
            text_config_key="text_config",
            use_cpu_initialization=False,
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            qk_layernorm=True,
            attention_output_gate=True,
        )

    def _get_gptmodel_args(self) -> dict:
        text_config = self.text_config
        rope_parameters = getattr(text_config, "rope_parameters", None) or getattr(text_config, "rope_scaling", None) or {}
        return dict(
            vocab_size=text_config.vocab_size,
            max_sequence_length=text_config.max_position_embeddings,
            position_embedding_type="rope",
            rotary_base=rope_parameters.get("rope_theta", getattr(text_config, "rope_theta", 10000)),
        )

    def _weight_to_mcore_format(
        self, mcore_weights_name: str, hf_weights: list[torch.Tensor]
    ) -> torch.Tensor:
        if "self_attention.linear_qkv." in mcore_weights_name and "layer_norm" not in mcore_weights_name:
            assert len(hf_weights) == 3
            text_config = self.text_config
            num_key_value_heads = text_config.num_key_value_heads
            hidden_dim = text_config.hidden_size
            num_attention_heads = text_config.num_attention_heads
            num_queries_per_group = num_attention_heads // num_key_value_heads
            head_dim = getattr(text_config, "head_dim", hidden_dim // num_attention_heads)
            group_dim = head_dim * num_attention_heads // num_key_value_heads
            q, k, v = hf_weights
            real_num_key_value_heads = q.shape[0] // (2 * group_dim)
            q = (
                q.view([real_num_key_value_heads, num_queries_per_group, 2, head_dim, -1])
                .transpose(1, 2)
                .flatten(1, 3)
            )
            k = k.view([real_num_key_value_heads, head_dim, -1])
            v = v.view([real_num_key_value_heads, head_dim, -1])
            out_shape = [-1, hidden_dim] if ".bias" not in mcore_weights_name else [-1]
            return torch.cat([q, k, v], dim=1).view(*out_shape).contiguous()

        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)
