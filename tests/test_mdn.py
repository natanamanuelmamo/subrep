"""Validation tests for the SubRep Motive Decomposition Network."""

import pytest
import torch

from generator.mdn import MotiveDecompositionNetwork


def test_mdn_single_input_shape():
    """Single context inputs should preserve unbatched output shapes."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    context = torch.randn(8)

    weight_params, support_values = model(context)

    assert weight_params.shape == (2,)
    assert support_values.shape == (2,)


def test_mdn_batched_input_shape():
    """Batched context inputs should preserve the batch dimension."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    context = torch.randn(5, 8)

    weight_params, support_values = model(context)

    assert weight_params.shape == (5, 2)
    assert support_values.shape == (5, 2)


def test_mdn_dirichlet_alpha_parameters_are_strictly_positive():
    """Dirichlet alpha parameters must always be strictly positive."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    context = torch.randn(5, 8)

    weight_params, _ = model(context)

    assert torch.all(weight_params > 0)


def test_mdn_support_values_are_non_negative():
    """Support values should be non-negative under the current support-function contract."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    context = torch.randn(5, 8)

    _, support_values = model(context)

    assert torch.all(support_values >= 0)


def test_mdn_two_objective_support_values_are_feasible_for_single_context():
    """Two-objective support values should define a non-empty W_x interval."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork(num_objectives=2)
    context = torch.randn(8)

    _, support_values = model.forward_inference(context)

    assert support_values.shape == (2,)
    assert torch.all(support_values >= 0)
    assert torch.all(support_values <= 1)
    assert torch.sum(support_values) >= 1.0


def test_mdn_two_objective_support_values_are_feasible_for_batched_contexts():
    """Batched two-objective support values should all define non-empty W_x intervals."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork(num_objectives=2)
    context = torch.randn(5, 8)

    _, support_values = model.forward_inference(context)

    assert support_values.shape == (5, 2)
    assert torch.all(support_values >= 0)
    assert torch.all(support_values <= 1)
    assert torch.all(torch.sum(support_values, dim=-1) >= 1.0)


@pytest.mark.parametrize("num_objectives", [2, 3, 5, 10, 50])
def test_mdn_support_values_feasible_for_any_M(num_objectives: int):
    """SASP must produce feasible support values for any objective count."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork(num_objectives=num_objectives)
    context = torch.randn(64, 8) * 10.0

    _, support_values = model.forward_inference(context)

    assert support_values.shape == (64, num_objectives)
    assert torch.all(support_values >= 0)
    assert torch.all(support_values <= 1)
    assert torch.all(torch.sum(support_values, dim=-1) >= 1.0 - 1e-6)


def test_sasp_hard_guarantee_with_extreme_logits():
    """SASP feasibility must hold even for extreme raw logits."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork(num_objectives=5)

    for scale in (1.0, 10.0, 50.0):
        raw = torch.randn(128, 10) * scale
        support_values = model._support_values_from_raw(raw)
        assert torch.all(support_values >= 0)
        assert torch.all(support_values <= 1)
        assert torch.all(torch.sum(support_values, dim=-1) >= 1.0 - 1e-6)


def test_sasp_is_permutation_equivariant():
    """Permuting objective indices must permute SASP outputs identically."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork(num_objectives=5)
    raw = torch.randn(16, 10)
    perm = torch.randperm(5)
    permuted_raw = torch.cat(
        [raw[..., :5][..., perm], raw[..., 5:][..., perm]],
        dim=-1,
    )

    support_values = model._support_values_from_raw(raw)
    permuted_support_values = model._support_values_from_raw(permuted_raw)

    assert torch.allclose(support_values[..., perm], permuted_support_values, atol=1e-6)


def test_sasp_surjectivity_round_trip():
    """Every feasible target must be reconstructible via analytic SASP witnesses."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork(num_objectives=4)

    for _ in range(32):
        target = torch.rand(4)
        target = target / target.sum() * torch.empty(()).uniform_(1.0, 1.8)
        target = torch.clamp(target, min=1e-4, max=1.0 - 1e-4)

        total = target.sum()
        base_allocation = target / total
        slack_gate = (target - base_allocation) / (1.0 - base_allocation)
        slack_gate = torch.clamp(slack_gate, min=1e-6, max=1.0 - 1e-6)

        base_logits = torch.log(base_allocation)
        slack_logits = torch.log(slack_gate / (1.0 - slack_gate))
        raw = torch.cat([base_logits, slack_logits])

        reconstructed = model._support_values_from_raw(raw.unsqueeze(0)).squeeze(0)
        assert torch.allclose(reconstructed, target, atol=1e-5)


def test_mdn_rejects_invalid_slack_floor():
    """slack_floor must lie in [0, 1)."""
    with pytest.raises(ValueError, match="slack_floor"):
        MotiveDecompositionNetwork(slack_floor=1.0)


def test_mdn_outputs_are_finite():
    """Both heads should produce finite tensors without NaN or Inf values."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    context = torch.randn(5, 8)

    weight_params, support_values = model(context)

    assert torch.isfinite(weight_params).all()
    assert torch.isfinite(support_values).all()


def test_mdn_synthetic_gradient_flow_reaches_parameters_and_input():
    """A synthetic combined loss should backpropagate through both heads and input.
    """
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    context = torch.randn(5, 8, requires_grad=True)

    weight_params, support_values = model(context)
    loss = weight_params.sum() + support_values.sum()
    loss.backward()

    assert context.grad is not None
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_mdn_rejects_invalid_input_dimension():
    """Model should raise a clear error when the feature dimension is wrong."""
    model = MotiveDecompositionNetwork()
    context = torch.randn(7)

    with pytest.raises(ValueError, match=r"Expected single context shape \(8,\)"):
        model(context)


def test_mdn_heads_are_independent_modules():
    """Distribution and support predictions must come from separate heads."""
    model = MotiveDecompositionNetwork()

    assert hasattr(model, "distribution_head")
    assert hasattr(model, "support_head")
    assert model.distribution_head is not model.support_head
