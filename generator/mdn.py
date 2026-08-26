# Motive Decomposition Network (MDN)

from __future__ import annotations

import torch
from torch import Tensor, nn


class MotiveDecompositionNetwork(nn.Module):
    """Shared-backbone MDN for runtime inference and auxiliary training.
    """
    def __init__(
        self,
        input_dim: int = 8,
        num_objectives: int = 2,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        alpha_epsilon: float = 1e-6,
        num_skills: int = 128,
        skill_embedding_dim: int = 8,
        slack_floor: float = 0.02,
    ) -> None:
        super().__init__()

        if num_hidden_layers < 1:
            raise ValueError(
                f"Expected num_hidden_layers >= 1, got {num_hidden_layers}"
            )
        if num_skills <= 0:
            raise ValueError(f"Expected num_skills > 0, got {num_skills}")
        if not (0.0 <= slack_floor < 1.0):
            raise ValueError(f"slack_floor must be in [0, 1), got {slack_floor}")

        self.input_dim = input_dim
        self.num_objectives = num_objectives
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.alpha_epsilon = alpha_epsilon
        self.num_skills = num_skills
        self.skill_embedding_dim = skill_embedding_dim
        self.slack_floor = float(slack_floor)

        trunk_layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(num_hidden_layers - 1):
            trunk_layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )

        self.trunk = nn.Sequential(*trunk_layers)
        self.distribution_head = nn.Linear(hidden_dim, num_objectives)
        # SASP: first M logits = base allocation, last M = slack gates.
        self.support_head = nn.Linear(hidden_dim, 2 * num_objectives)
        self.skill_embedding = nn.Embedding(num_skills, skill_embedding_dim)
        self.auxiliary_fusion = nn.Sequential(
            nn.Linear(hidden_dim + skill_embedding_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gate_head = nn.Linear(hidden_dim, 1)
        self.motive_head = nn.Linear(hidden_dim, num_objectives)
        self.softplus = nn.Softplus()

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply stable initialization across all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.skill_embedding.weight, mean=0.0, std=0.02)

    def _encode_context(self, context: Tensor) -> tuple[Tensor, bool]:
        if context.ndim not in (1, 2):
            raise ValueError(
                f"Expected context with shape ({self.input_dim},) or "
                f"(N, {self.input_dim}), got tensor with shape {tuple(context.shape)}"
            )

        is_single_input = context.ndim == 1
        if is_single_input:
            if context.shape[0] != self.input_dim:
                raise ValueError(
                    f"Expected single context shape ({self.input_dim},), "
                    f"got {tuple(context.shape)}"
                )
            context = context.unsqueeze(0)
        elif context.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected batched context shape (N, {self.input_dim}), "
                f"got {tuple(context.shape)}"
            )

        features = self.trunk(context)
        return features, is_single_input

    def _support_values_from_raw(self, raw_support: Tensor) -> Tensor:
        """Softmax-Anchored Slack Parameterization (SASP).

        s_i = p_i + (1 - p_i) * g_i
          p = softmax(base logits)              -> sum(p) == 1, p_i in (0, 1)
          g = g_min + (1 - g_min) * sigmoid(logits) -> g_i in (g_min, 1)

        Guarantees, for any num_objectives M and any weights:
          0 < s_i < 1  and  sum(s) >= 1.
        """
        num_objectives = self.num_objectives
        if raw_support.shape[-1] != 2 * num_objectives:
            raise ValueError(
                f"support head must emit 2*M={2 * num_objectives} logits, "
                f"got {raw_support.shape[-1]}"
            )

        base_logits = raw_support[..., :num_objectives]
        slack_logits = raw_support[..., num_objectives:]
        base_allocation = torch.softmax(base_logits, dim=-1)
        slack_gate = self.slack_floor + (1.0 - self.slack_floor) * torch.sigmoid(
            slack_logits
        )
        return base_allocation + (1.0 - base_allocation) * slack_gate

    def forward_inference(self, context: Tensor) -> tuple[Tensor, Tensor]:
        features, is_single_input = self._encode_context(context)
        weight_params = self.softplus(self.distribution_head(features)) + self.alpha_epsilon
        support_values = self._support_values_from_raw(self.support_head(features))

        if is_single_input:
            weight_params = weight_params.squeeze(0)
            support_values = support_values.squeeze(0)

        return weight_params, support_values

    def forward_auxiliary(self, context: Tensor, skill_id: Tensor) -> tuple[Tensor, Tensor]:
        features, is_single_input = self._encode_context(context)

        if is_single_input:
            if skill_id.ndim != 0:
                raise ValueError(f"Expected scalar skill_id for single context, got shape {tuple(skill_id.shape)}")
            skill_id = skill_id.unsqueeze(0)
        else:
            if skill_id.ndim != 1:
                raise ValueError(f"Expected batched skill_id shape (N,), got shape {tuple(skill_id.shape)}")
            if skill_id.shape[0] != features.shape[0]:
                raise ValueError("skill_id batch size must match context batch size")

        embedded_skill = self.skill_embedding(skill_id.long())
        fused = self.auxiliary_fusion(torch.cat([features, embedded_skill], dim=-1))
        gate_logits = self.gate_head(fused).squeeze(-1)
        q_hat = self.motive_head(fused)

        if is_single_input:
            gate_logits = gate_logits.squeeze(0)
            q_hat = q_hat.squeeze(0)

        return gate_logits, q_hat

    def forward(self, context: Tensor) -> tuple[Tensor, Tensor]:
        return self.forward_inference(context)
