"""
RATIO multi-head detector.

The model uses a DeBERTa-v3 masked-language-model backbone as the shared text
encoder and adds two task heads:

1. a six-way classification head for the AI involvement ratio bins
2. a sigmoid regression head for the continuous LIR estimate

The optional auxiliary head and metadata features are retained for compatibility
with archived experiments, but the public RATIO training path does not enable
metadata inputs by default.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

if TYPE_CHECKING:
    from transformers import AutoTokenizer


DEFAULT_AUX_TARGET_NAMES: tuple[str, ...] = (
    "lir",
    "jaccard",
    "sentence_jaccard",
    "cosine",
)


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU) activation.

    GLU(x) = x * sigmoid(W1 @ x + b1)
    where x * sigmoid(W2 @ x + b2) is element-wise multiplied.

    This is more expressive than plain GELU and helps the model
    learn better representations for classification.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.fc2(x))
        return self.dropout(self.fc1(x) * gate)


class MultiHeadDetector(nn.Module):
    """
    Multi-task detector for AI text proportion estimation.

    Args:
        model_name: HuggingFace model name or local path for DeBERTa-v3-large
        num_classes: Number of classification bins (default: 6 for 0-100% in 20% steps)
        dropout: Dropout probability for classification/regression heads
        pll_sample_ratio: Retained for compatibility with older checkpoints.
        use_continuous_features: Whether to use continuous features
            (jaccard, sentence_jaccard, cosine, lir) as auxiliary input.
            These fields are disabled by default because benchmark data
            derives them from original_text/mixed_text pairs, so they are
            answer-correlated and should only be used for ablations.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-large",
        num_classes: int = 6,
        dropout: float = 0.1,
        pll_sample_ratio: float = 0.15,
        use_continuous_features: bool = False,
        aux_target_names: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pll_sample_ratio = pll_sample_ratio
        self.use_continuous_features = use_continuous_features
        self.aux_target_names = tuple(aux_target_names or DEFAULT_AUX_TARGET_NAMES)
        self.num_aux_targets = len(self.aux_target_names)

        self.mlm_model = AutoModelForMaskedLM.from_pretrained(model_name)
        self._fix_mlm_head(model_name)

        hidden_size: int = self.mlm_model.config.hidden_size
        self.pool_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.pool_ln = nn.LayerNorm(hidden_size)

        # 连续特征编码器（4个特征：jaccard, sentence_jaccard, cosine, lir）
        cont_encoded_dim = 32
        if use_continuous_features:
            self.cont_feature_dim = 4
            self.cont_fc1 = nn.Linear(self.cont_feature_dim, cont_encoded_dim)
            self.cont_ln = nn.LayerNorm(cont_encoded_dim)
            self.cont_fc2 = nn.Linear(cont_encoded_dim, hidden_size)  # 32 -> 1024
        else:
            self.cont_feature_dim = 0
            cont_encoded_dim = 0

        # 分类头输入维度 = hidden_size (连续特征通过另一条路径传入)
        # Classification head: Enhanced with residual connections and GLU
        # 输入维度：1024 (文本) + 1024 (连续特征) = 2048
        cls_head_input_dim = hidden_size * 2 if use_continuous_features else hidden_size
        self.cls_ln = nn.LayerNorm(cls_head_input_dim)
        self.cls_glu = GatedLinearUnit(cls_head_input_dim, cls_head_input_dim // 2, dropout=dropout)
        self.cls_ln2 = nn.LayerNorm(cls_head_input_dim // 2)
        self.cls_fc_out = nn.Linear(cls_head_input_dim // 2, num_classes)
        self.cls_dropout = nn.Dropout(dropout)

        # 回归头：使用 Sigmoid 做平滑边界约束，避免 hard clamp 截断梯度
        reg_head_input_dim = hidden_size * 2 if use_continuous_features else hidden_size
        self.reg_ln = nn.LayerNorm(reg_head_input_dim)
        self.reg_glu = GatedLinearUnit(reg_head_input_dim, reg_head_input_dim // 4, dropout=dropout)
        self.reg_ln2 = nn.LayerNorm(reg_head_input_dim // 4)
        self.reg_fc_out = nn.Linear(reg_head_input_dim // 4, 1)
        self.reg_dropout = nn.Dropout(dropout)

        self.aux_ln = nn.LayerNorm(reg_head_input_dim)
        self.aux_glu = GatedLinearUnit(reg_head_input_dim, reg_head_input_dim // 4, dropout=dropout)
        self.aux_ln2 = nn.LayerNorm(reg_head_input_dim // 4)
        self.aux_fc_out = nn.Linear(reg_head_input_dim // 4, self.num_aux_targets)
        self.aux_dropout = nn.Dropout(dropout)

        # 权重初始化
        self._init_heads()

        self._mask_token_id: int = 0
        self._special_ids: set[int] = set()

    def _init_heads(self) -> None:
        """Initialize task heads with proper weight initialization for stability."""
        # Use smaller gains for GLU-based initialization
        for module in [
            self.pool_proj,
            self.cls_glu.fc1,
            self.cls_glu.fc2,
            self.reg_glu.fc1,
            self.reg_glu.fc2,
            self.aux_glu.fc1,
            self.aux_glu.fc2,
        ]:
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.cls_fc_out.weight, gain=0.5)
        nn.init.zeros_(self.cls_fc_out.bias)

        nn.init.xavier_uniform_(self.reg_fc_out.weight, gain=0.01)  # Small init for output
        nn.init.zeros_(self.reg_fc_out.bias)

        nn.init.xavier_uniform_(self.aux_fc_out.weight, gain=0.01)
        nn.init.zeros_(self.aux_fc_out.bias)

    # ------------------------------------------------------------------ #
    #  MLM head weight fix (for models with non-standard key names)      #
    # ------------------------------------------------------------------ #
    def _fix_mlm_head(self, model_name: str) -> None:
        """Remap lm_predictions.lm_head.* → cls.predictions.* for model compatibility."""
        try:
            if os.path.isdir(model_name):
                sf = os.path.join(model_name, "model.safetensors")
                pt = os.path.join(model_name, "pytorch_model.bin")
                if os.path.exists(sf):
                    from safetensors.torch import load_file
                    raw_sd = load_file(sf)
                elif os.path.exists(pt):
                    raw_sd = torch.load(pt, map_location="cpu", weights_only=True)
                else:
                    return
            else:
                from huggingface_hub import hf_hub_download

                try:
                    from safetensors.torch import load_file
                    path = hf_hub_download(model_name, "model.safetensors", local_files_only=True)
                    raw_sd = load_file(path)
                except Exception:
                    path = hf_hub_download(model_name, "pytorch_model.bin", local_files_only=True)
                    raw_sd = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            return

        if not any(k.startswith("lm_predictions.") for k in raw_sd):
            return

        KEY_MAP = {
            "lm_predictions.lm_head.dense.weight": "cls.predictions.transform.dense.weight",
            "lm_predictions.lm_head.dense.bias": "cls.predictions.transform.dense.bias",
            "lm_predictions.lm_head.LayerNorm.weight": "cls.predictions.transform.LayerNorm.weight",
            "lm_predictions.lm_head.LayerNorm.bias": "cls.predictions.transform.LayerNorm.bias",
            "lm_predictions.lm_head.bias": "cls.predictions.bias",
        }

        sd = self.mlm_model.state_dict()
        patched = False
        for src, dst in KEY_MAP.items():
            if src in raw_sd and dst in sd:
                sd[dst] = raw_sd[src]
                patched = True

        if "cls.predictions.decoder.bias" in sd and "lm_predictions.lm_head.bias" in raw_sd:
            sd["cls.predictions.decoder.bias"] = raw_sd["lm_predictions.lm_head.bias"]
            patched = True

        if patched:
            self.mlm_model.load_state_dict(sd, strict=False)
            print("[FIX] Remapped MLM head weights "
                  "(lm_predictions.* → cls.predictions.*)")

    # ------------------------------------------------------------------ #
    #  Tokenizer binding                                                  #
    # ------------------------------------------------------------------ #
    def set_special_token_ids(self, tokenizer: AutoTokenizer) -> None:
        """Cache special-token IDs so masking operations skip them."""
        self._special_ids = set()
        for attr in (
            "cls_token_id", "sep_token_id", "pad_token_id",
            "bos_token_id", "eos_token_id",
        ):
            tid = getattr(tokenizer, attr, None)
            if tid is not None:
                self._special_ids.add(tid)
        self._mask_token_id = tokenizer.mask_token_id

    # ------------------------------------------------------------------ #
    #  Classification + Regression forward                               #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        continuous_features: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the multi-head detector.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_length)
            attention_mask: Attention mask of shape (batch_size, seq_length)
            continuous_features: Optional continuous features of shape (batch_size, 4)
                               [jaccard, sentence_jaccard, cosine, lir]
            return_aux: When True, also return continuous metric predictions
                [lir, jaccard, sentence_jaccard, cosine].

        Returns:
            cls_logits: Classification logits of shape (batch_size, num_classes)
            reg_output: Regression output of shape (batch_size,), values in [0, 1]
            aux_output: Optional tensor of shape (batch_size, num_aux_targets)
        """
        hidden_states = self.mlm_model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        head_dtype = self.pool_proj.weight.dtype
        if hidden_states.dtype != head_dtype:
            hidden_states = hidden_states.to(dtype=head_dtype)

        cls_token = hidden_states[:, 0]  # [CLS] token representation
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        mean_pool = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        cls_token = self.pool_ln(self.pool_proj(torch.cat([cls_token, mean_pool], dim=-1)))
        cls_token = F.gelu(cls_token)

        # 融合连续特征
        if self.use_continuous_features:
            if continuous_features is None:
                continuous_features = torch.zeros(
                    cls_token.size(0),
                    self.cont_feature_dim,
                    device=cls_token.device,
                    dtype=cls_token.dtype,
                )
            else:
                continuous_features = continuous_features.to(device=cls_token.device, dtype=cls_token.dtype)
            # 将连续特征编码后拼接到 cls_token
            cont_encoded = self.cont_ln(self.cont_fc1(continuous_features))
            cont_encoded = F.gelu(cont_encoded)
            cont_encoded = self.cont_fc2(cont_encoded)  # 32 -> 1024
            # 拼接而不是相加（更稳定的融合方式）
            cls_features = torch.cat([cls_token, cont_encoded], dim=-1)
        else:
            cls_features = cls_token

        # Classification head: LN -> GLU -> LN -> Output
        cls_normed = self.cls_ln(cls_features)
        cls_glu_out = self.cls_glu(cls_normed)
        cls_glu_out = self.cls_ln2(cls_glu_out)
        cls_logits = self.cls_fc_out(cls_glu_out)

        # Regression head: LN -> GLU -> LN -> Output
        reg_normed = self.reg_ln(cls_features)
        reg_glu_out = self.reg_glu(reg_normed)
        reg_glu_out = self.reg_ln2(reg_glu_out)
        reg_output = torch.sigmoid(self.reg_fc_out(reg_glu_out).squeeze(-1))

        if return_aux:
            aux_normed = self.aux_ln(cls_features)
            aux_glu_out = self.aux_glu(aux_normed)
            aux_glu_out = self.aux_ln2(aux_glu_out)
            aux_output = torch.sigmoid(self.aux_fc_out(aux_glu_out))
            return cls_logits, reg_output, aux_output

        return cls_logits, reg_output

    # ------------------------------------------------------------------ #
    #  Utility methods                                                    #
    # ------------------------------------------------------------------ #
    def get_config(self) -> dict:
        """Return model configuration as a dictionary."""
        return {
            "num_classes": self.num_classes,
            "pll_sample_ratio": self.pll_sample_ratio,
            "use_continuous_features": self.use_continuous_features,
            "aux_target_names": list(self.aux_target_names),
            "hidden_size": self.mlm_model.config.hidden_size,
        }
