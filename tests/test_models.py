"""Tests for CompositionalMLP and SetTransformerModel — shapes, grads, padding."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.compositional_mlp import CompositionalMLP
from src.models.set_transformer import SetTransformerModel


# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------

B = 4
N = 10
D_COMP = 50
D_SCEN = 40
N_COMPONENTS = 113
N_TYPES = 10
N_MODES = 10


def _random_component_features():
    x = torch.randn(B, N, D_COMP)
    x[..., 0] = torch.randint(0, N_COMPONENTS, (B, N)).float()
    x[..., 1] = torch.randint(0, N_TYPES, (B, N)).float()
    return x


def _random_scenario_features():
    x = torch.randn(B, D_SCEN)
    x[:, 0] = torch.randint(0, N_MODES, (B,)).float()
    return x


def _random_mask(pad: int = 2):
    mask = torch.ones(B, N)
    if pad > 0:
        mask[:, -pad:] = 0
    return mask


# ---------------------------------------------------------------------------
# CompositionalMLP
# ---------------------------------------------------------------------------


def test_compositional_mlp_forward():
    model = CompositionalMLP(n_modes=N_MODES, d_scen_other=D_SCEN - 1)
    scen = _random_scenario_features()
    out = model(scen)
    assert set(out.keys()) == {"reg_out", "sign_logit"}
    assert out["reg_out"].shape == (B, 2)
    assert out["sign_logit"].shape == (B, 1)
    assert not torch.isnan(out["reg_out"]).any()


def test_compositional_mlp_gradient_flow():
    model = CompositionalMLP(n_modes=N_MODES, d_scen_other=D_SCEN - 1)
    scen = _random_scenario_features()
    out = model(scen)
    (out["reg_out"].pow(2).mean() + out["sign_logit"].pow(2).mean()).backward()
    any_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    assert any_grad


# ---------------------------------------------------------------------------
# SetTransformerModel
# ---------------------------------------------------------------------------


def _build_st():
    return SetTransformerModel(
        n_components=N_COMPONENTS,
        n_types=N_TYPES,
        n_modes=N_MODES,
        d_comp_numeric=D_COMP - 2,
        d_scen_other=D_SCEN - 1,
        dropout=0.0,
    )


def test_set_transformer_forward():
    model = _build_st().eval()
    comp = _random_component_features()
    mask = _random_mask()
    scen = _random_scenario_features()
    out = model(comp, mask, scen)
    assert set(out.keys()) == {"reg_out", "sign_logit", "component_attention"}
    assert out["reg_out"].shape == (B, 2)
    assert out["sign_logit"].shape == (B, 1)
    assert out["component_attention"] is not None
    assert out["component_attention"].shape == (B, 4, 1, N)  # num_heads=4
    assert not torch.isnan(out["reg_out"]).any()
    assert not torch.isnan(out["component_attention"]).any()


def test_set_transformer_gradient_flow():
    model = _build_st()
    comp = _random_component_features()
    mask = _random_mask()
    scen = _random_scenario_features()
    out = model(comp, mask, scen)
    loss = out["reg_out"].pow(2).mean() + out["sign_logit"].pow(2).mean()
    loss.backward()
    any_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    assert any_grad


def test_set_transformer_padding_invariance():
    """Scrambling padded-component features must not change the predictions."""
    torch.manual_seed(7)
    model = _build_st().eval()
    comp = _random_component_features()
    mask = _random_mask(pad=3)  # last 3 positions padded
    scen = _random_scenario_features()

    out_a = model(comp, mask, scen)

    comp2 = comp.clone()
    comp2[:, -3:, 2:] = torch.randn(B, 3, D_COMP - 2) * 50
    # categorical indices in padded slots shouldn't matter either
    comp2[:, -3:, 0] = torch.randint(0, N_COMPONENTS, (B, 3)).float()
    comp2[:, -3:, 1] = torch.randint(0, N_TYPES, (B, 3)).float()
    out_b = model(comp2, mask, scen)

    torch.testing.assert_close(out_a["reg_out"], out_b["reg_out"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_a["sign_logit"], out_b["sign_logit"], atol=1e-5, rtol=1e-5)


def test_set_transformer_n_params_rough_budget():
    model = _build_st()
    n = sum(p.numel() for p in model.parameters())
    # expect ~60-120K depending on embedding vocab sizes; be permissive
    assert 30_000 < n < 200_000, f"unexpected param count {n}"
