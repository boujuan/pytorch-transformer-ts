"""
Tests for TACTiS-2 two-stage hyperparameter tuning via get_params().

Validates that:
  - Phase 1 (Stage 1) returns only marginal/flow/decoder params + common params
  - Phase 2 (Stage 2) returns only copula/ac_mlp params + common params
  - Phase 2 with fixed Stage 1 params uses those fixed values for common params
  - Phase 0 (legacy) returns ALL params
  - skip_copula propagates correctly through the model hierarchy
"""

import pytest
from optuna.trial import FixedTrial

from pytorch_transformer_ts.tactis_2.estimator import TACTiS2Estimator
from pytorch_transformer_ts.tactis_2.module import TACTiS2Model


# ---------------------------------------------------------------------------
# Helper: build a FixedTrial param dict covering every suggest_* call
# ---------------------------------------------------------------------------

# Parameters used only in Stage 1 (marginal/flow/decoder)
STAGE1_ONLY_PARAMS = {
    "stage1_activation_function": "relu",
    "marginal_embedding_dim_per_head": 32,
    "marginal_num_heads": 4,
    "marginal_num_layers": 3,
    "flow_input_encoder_layers": 3,
    "flow_series_embedding_dim": 16,
    "decoder_dsf_num_layers": 2,
    "decoder_dsf_hidden_dim": 64,
    "decoder_mlp_num_layers": 3,
    "decoder_mlp_hidden_dim": 32,
    "decoder_transformer_num_layers": 2,
    "decoder_transformer_embedding_dim_per_head": 64,
    "decoder_transformer_num_heads": 3,
    "decoder_num_bins": 100,
    "lr_stage1": 1e-4,
    "weight_decay_stage1": 1e-6,
    "gradient_clip_val_stage1": 1.0,
    "eta_min_fraction_s1": 0.005,
    "bagging_size": None,
}

# Parameters used only in Stage 2 (copula/ac_mlp)
STAGE2_ONLY_PARAMS = {
    "stage2_start_epoch": 10,  # Only tuned in Phase 2 (controls when copula training begins)
    "stage2_activation_function": "relu",
    "copula_embedding_dim_per_head": 32,
    "copula_num_heads": 4,
    "copula_num_layers": 2,
    "copula_input_encoder_layers": 3,
    "copula_series_embedding_dim": 64,
    "ac_mlp_num_layers": 3,
    "ac_mlp_dim": 128,
    "lr_stage2": 1e-4,
    "weight_decay_stage2": 1e-5,
    "gradient_clip_val_stage2": 5.0,
    "eta_min_fraction_s2": 0.005,
}

# Common parameters shared across phases (no stage2_start_epoch — it's phase-specific)
COMMON_PARAMS = {
    "context_length_factor": 10,
    "encoder_type": "standard",
    "batch_size": 64,
    "dropout_rate": 0.01,
    "loss_normalization": "series",
}

# Keys that identify common params in the output dict
COMMON_PARAM_KEYS = set(COMMON_PARAMS.keys())

# Keys that identify Stage 1 specific params in the output dict
STAGE1_PARAM_KEYS = set(STAGE1_ONLY_PARAMS.keys())

# Keys that identify Stage 2 specific params in the output dict
STAGE2_PARAM_KEYS = set(STAGE2_ONLY_PARAMS.keys())


def _all_fixed_params():
    """Return a merged dict of ALL possible param suggestions with reasonable defaults."""
    return {**COMMON_PARAMS, **STAGE1_ONLY_PARAMS, **STAGE2_ONLY_PARAMS}


def _phase1_fixed_params():
    """Return param dict needed for a Phase 1 FixedTrial."""
    return {**COMMON_PARAMS, **STAGE1_ONLY_PARAMS}


def _phase2_fixed_params():
    """Return param dict needed for a Phase 2 FixedTrial."""
    return {**COMMON_PARAMS, **STAGE2_ONLY_PARAMS}


# ---------------------------------------------------------------------------
# Test 1: Phase 1 returns Stage 1 + common params only
# ---------------------------------------------------------------------------

class TestGetParamsPhase1:
    """Verify get_params(trial, tuning_phase=1) returns Stage 1 params only."""

    def test_get_params_phase1_returns_stage1_only(self):
        """
        Phase 1 must return:
          - All common params (context_length_factor, encoder_type, batch_size, etc.)
          - All Stage 1 params (marginal_*, flow_*, decoder_*, lr_stage1, etc.)
        Phase 1 must NOT return:
          - Any copula_* params
          - Any ac_mlp_* params
          - lr_stage2
        """
        trial = FixedTrial(_phase1_fixed_params())
        result = TACTiS2Estimator.get_params(trial, tuning_phase=1)

        # -- Verify all common params are present --
        for key in COMMON_PARAM_KEYS:
            assert key in result, f"Common param '{key}' missing from Phase 1 result"

        # -- Verify all Stage 1 params are present --
        for key in STAGE1_PARAM_KEYS:
            assert key in result, f"Stage 1 param '{key}' missing from Phase 1 result"

        # -- Verify lr_stage1 is present --
        assert "lr_stage1" in result, "lr_stage1 must be in Phase 1 result"

        # -- Verify Stage 2 params are absent (except stage2_start_epoch which is forced to 9999) --
        assert "lr_stage2" not in result, "lr_stage2 must NOT be in Phase 1 result"
        for key in STAGE2_PARAM_KEYS:
            if key == "stage2_start_epoch":
                continue  # Allowed in Phase 1 as a forced override (9999)
            assert key not in result, f"Stage 2 param '{key}' must NOT be in Phase 1 result"

        # -- Verify Phase 1 disables copula and prevents stage transition --
        assert result["skip_copula"] is True, "Phase 1 must set skip_copula=True"
        assert result["lock_skip_copula"] is True, "Phase 1 must set lock_skip_copula=True"
        assert result["stage2_start_epoch"] == 9999, "Phase 1 must prevent stage transition"

    def test_phase1_values_match_trial_suggestions(self):
        """Returned values must match what the FixedTrial provides."""
        fixed = _phase1_fixed_params()
        trial = FixedTrial(fixed)
        result = TACTiS2Estimator.get_params(trial, tuning_phase=1)

        for key, expected in fixed.items():
            assert result[key] == expected, (
                f"Phase 1 param '{key}': expected {expected!r}, got {result[key]!r}"
            )


# ---------------------------------------------------------------------------
# Test 2: Phase 2 returns Stage 2 + common params only
# ---------------------------------------------------------------------------

class TestGetParamsPhase2:
    """Verify get_params(trial, tuning_phase=2) returns Stage 2 params only."""

    def test_get_params_phase2_returns_stage2_only(self):
        """
        Phase 2 must return:
          - All common params
          - All Stage 2 params (copula_*, ac_mlp_*, lr_stage2, etc.)
        Phase 2 must NOT return:
          - Any marginal_* params
          - Any flow_* params (flow_input_encoder_layers, flow_series_embedding_dim)
          - Any decoder_dsf_*, decoder_mlp_*, decoder_transformer_* params
          - lr_stage1
        """
        trial = FixedTrial(_phase2_fixed_params())
        result = TACTiS2Estimator.get_params(trial, tuning_phase=2)

        # -- Verify all common params are present --
        for key in COMMON_PARAM_KEYS:
            assert key in result, f"Common param '{key}' missing from Phase 2 result"

        # -- Verify all Stage 2 params are present --
        for key in STAGE2_PARAM_KEYS:
            assert key in result, f"Stage 2 param '{key}' missing from Phase 2 result"

        # -- Verify lr_stage2 is present --
        assert "lr_stage2" in result, "lr_stage2 must be in Phase 2 result"

        # -- Verify Stage 1 params are absent --
        assert "lr_stage1" not in result, "lr_stage1 must NOT be in Phase 2 result"
        for key in STAGE1_PARAM_KEYS:
            assert key not in result, f"Stage 1 param '{key}' must NOT be in Phase 2 result"

    def test_phase2_values_match_trial_suggestions(self):
        """Returned values must match what the FixedTrial provides."""
        fixed = _phase2_fixed_params()
        trial = FixedTrial(fixed)
        result = TACTiS2Estimator.get_params(trial, tuning_phase=2)

        for key, expected in fixed.items():
            assert result[key] == expected, (
                f"Phase 2 param '{key}': expected {expected!r}, got {result[key]!r}"
            )


# ---------------------------------------------------------------------------
# Test 3: Phase 2 with fixed Stage 1 params
# ---------------------------------------------------------------------------

class TestGetParamsPhase2WithFixedStage1:
    """
    Verify get_params(trial, tuning_phase=2, dynamic_kwargs={"stage1_fixed_params": {...}})
    uses fixed values for common params while Stage 2 params remain tunable.
    """

    def test_get_params_phase2_with_fixed_stage1_params(self):
        """
        When stage1_fixed_params is provided:
          - common_params must use the fixed values (NOT trial suggestions)
          - Stage 1 architecture params must be passed through (so the model
            is built with Phase 1's best marginal/flow/decoder config)
          - Stage 2 params must still come from trial suggestions
        """
        # Realistic Phase 1 best trial (common + stage1-specific params)
        stage1_fixed = {
            # Common params
            "context_length_factor": 20,
            "encoder_type": "temporal",
            "batch_size": 128,
            "dropout_rate": 0.007,
            "loss_normalization": "both",
            # Stage 1 architecture params (must be passed through to estimator)
            **STAGE1_ONLY_PARAMS,
        }

        # The FixedTrial only needs Stage 2 params since common + stage1 params
        # are taken from stage1_fixed_params (not from the trial)
        trial = FixedTrial(STAGE2_ONLY_PARAMS)

        result = TACTiS2Estimator.get_params(
            trial,
            tuning_phase=2,
            dynamic_kwargs={"stage1_fixed_params": stage1_fixed},
        )

        # -- Common params must equal the fixed values --
        for key in ("context_length_factor", "encoder_type", "batch_size", "dropout_rate", "loss_normalization"):
            assert result[key] == stage1_fixed[key], (
                f"Common param '{key}' should be fixed at {stage1_fixed[key]!r}, "
                f"got {result[key]!r}"
            )

        # -- Stage 1 architecture params must be passed through --
        for key in STAGE1_PARAM_KEYS:
            assert key in result, f"Stage 1 arch param '{key}' must be passed through in Phase 2"
            assert result[key] == STAGE1_ONLY_PARAMS[key], (
                f"Stage 1 arch param '{key}': expected {STAGE1_ONLY_PARAMS[key]!r}, got {result[key]!r}"
            )

        # -- Stage 2 params must still be tunable (from trial) --
        for key, expected in STAGE2_ONLY_PARAMS.items():
            assert key in result, f"Stage 2 param '{key}' missing from result"
            assert result[key] == expected, (
                f"Stage 2 param '{key}': expected {expected!r}, got {result[key]!r}"
            )

        # -- Phase 2 must enable copula --
        assert result["skip_copula"] is False, "Phase 2 must set skip_copula=False"

    def test_common_params_differ_from_defaults_when_fixed(self):
        """
        Ensure the fixed values actually override what a normal Phase 2 trial
        would have produced (guards against the fixed path being ignored).
        """
        # Use deliberately unusual common values + stage1 arch params
        stage1_fixed = {
            "context_length_factor": 25,
            "encoder_type": "temporal",
            "batch_size": 512,
            "dropout_rate": 0.015,
            "loss_normalization": "none",
            **STAGE1_ONLY_PARAMS,
        }

        trial = FixedTrial(STAGE2_ONLY_PARAMS)
        result = TACTiS2Estimator.get_params(
            trial,
            tuning_phase=2,
            dynamic_kwargs={"stage1_fixed_params": stage1_fixed},
        )

        # Each common param should exactly match stage1_fixed
        assert result["context_length_factor"] == 25
        assert result["encoder_type"] == "temporal"
        assert result["batch_size"] == 512
        assert result["dropout_rate"] == 0.015
        assert result["loss_normalization"] == "none"
        # stage2_start_epoch comes from trial (Phase 2 tunable), not from stage1_fixed
        assert result["stage2_start_epoch"] == STAGE2_ONLY_PARAMS["stage2_start_epoch"]
        # skip_copula must be False for Phase 2
        assert result["skip_copula"] is False


# ---------------------------------------------------------------------------
# Test 4: Phase 0 (legacy) returns ALL params
# ---------------------------------------------------------------------------

class TestGetParamsPhase0:
    """Verify get_params(trial, tuning_phase=0) returns all params."""

    def test_get_params_phase0_returns_all_params(self):
        """
        Phase 0 (legacy) must return:
          - All common params
          - All Stage 1 params (marginal_*, flow_*, decoder_*, lr_stage1, etc.)
          - All Stage 2 params (copula_*, ac_mlp_*, lr_stage2, etc.)
        """
        trial = FixedTrial(_all_fixed_params())
        result = TACTiS2Estimator.get_params(trial, tuning_phase=0)

        # -- Verify all common params --
        for key in COMMON_PARAM_KEYS:
            assert key in result, f"Common param '{key}' missing from Phase 0 result"

        # -- Verify all Stage 1 params (using keys from the legacy block) --
        stage1_keys_in_legacy = {
            "stage1_activation_function",
            "marginal_embedding_dim_per_head",
            "marginal_num_heads",
            "marginal_num_layers",
            "flow_input_encoder_layers",
            "flow_series_embedding_dim",
            "decoder_dsf_num_layers",
            "decoder_dsf_hidden_dim",
            "decoder_mlp_num_layers",
            "decoder_mlp_hidden_dim",
            "decoder_transformer_num_layers",
            "decoder_transformer_embedding_dim_per_head",
            "decoder_transformer_num_heads",
            "decoder_num_bins",
            "lr_stage1",
            "weight_decay_stage1",
            "gradient_clip_val_stage1",
            "eta_min_fraction_s1",
            "bagging_size",
        }
        for key in stage1_keys_in_legacy:
            assert key in result, f"Stage 1 param '{key}' missing from Phase 0 result"

        # -- Verify all Stage 2 params --
        stage2_keys_in_legacy = {
            "stage2_activation_function",
            "copula_embedding_dim_per_head",
            "copula_num_heads",
            "copula_num_layers",
            "copula_input_encoder_layers",
            "copula_series_embedding_dim",
            "ac_mlp_num_layers",
            "ac_mlp_dim",
            "lr_stage2",
            "weight_decay_stage2",
            "gradient_clip_val_stage2",
            "eta_min_fraction_s2",
        }
        for key in stage2_keys_in_legacy:
            assert key in result, f"Stage 2 param '{key}' missing from Phase 0 result"

        # -- Verify both lr values present --
        assert "lr_stage1" in result, "lr_stage1 must be in Phase 0 result"
        assert "lr_stage2" in result, "lr_stage2 must be in Phase 0 result"

    def test_phase0_contains_both_marginal_and_copula_params(self):
        """Explicit check that marginal_* and copula_* coexist in Phase 0."""
        trial = FixedTrial(_all_fixed_params())
        result = TACTiS2Estimator.get_params(trial, tuning_phase=0)

        marginal_keys = [k for k in result if k.startswith("marginal_")]
        copula_keys = [k for k in result if k.startswith("copula_")]

        assert len(marginal_keys) > 0, "Phase 0 must contain marginal_* params"
        assert len(copula_keys) > 0, "Phase 0 must contain copula_* params"

    def test_phase0_values_match_trial_suggestions(self):
        """Returned values must match what the FixedTrial provides."""
        fixed = _all_fixed_params()
        trial = FixedTrial(fixed)
        result = TACTiS2Estimator.get_params(trial, tuning_phase=0)

        for key, expected in fixed.items():
            assert result[key] == expected, (
                f"Phase 0 param '{key}': expected {expected!r}, got {result[key]!r}"
            )


# ---------------------------------------------------------------------------
# Test 5: skip_copula propagates through TACTiS2Model -> TACTiS
# ---------------------------------------------------------------------------

class TestSkipCopulaPropagation:
    """Verify skip_copula propagates from TACTiS2Model to the inner TACTiS instance."""

    @pytest.fixture
    def _minimal_model_kwargs(self):
        """Return the minimum kwargs needed to construct a TACTiS2Model."""
        return dict(
            num_series=3,
            context_length=10,
            prediction_length=5,
            flow_series_embedding_dim=8,
            copula_series_embedding_dim=8,
            flow_input_encoder_layers=1,
            copula_input_encoder_layers=1,
            marginal_embedding_dim_per_head=8,
            marginal_num_heads=2,
            marginal_num_layers=1,
            copula_embedding_dim_per_head=8,
            copula_num_heads=2,
            copula_num_layers=1,
            decoder_dsf_num_layers=1,
            decoder_dsf_hidden_dim=16,
            decoder_mlp_num_layers=1,
            decoder_mlp_hidden_dim=16,
            decoder_transformer_num_layers=1,
            decoder_transformer_embedding_dim_per_head=8,
            decoder_transformer_num_heads=2,
            decoder_num_bins=50,
            ac_mlp_num_layers=1,
            dropout_rate=0.1,
            num_parallel_samples=10,
        )

    def test_skip_copula_propagates_through_module(self, _minimal_model_kwargs):
        """
        When TACTiS2Model is created with skip_copula=True, the inner TACTiS
        instance (model.tactis) must also have skip_copula=True.
        """
        model = TACTiS2Model(skip_copula=True, **_minimal_model_kwargs)

        assert model.skip_copula is True, (
            "TACTiS2Model.skip_copula should be True"
        )
        assert model.tactis.skip_copula is True, (
            "TACTiS (inner model).skip_copula should be True when TACTiS2Model "
            "is created with skip_copula=True"
        )

    def test_skip_copula_false_propagates(self, _minimal_model_kwargs):
        """
        When TACTiS2Model is created with skip_copula=False, the inner TACTiS
        instance must also have skip_copula=False.
        """
        model = TACTiS2Model(skip_copula=False, **_minimal_model_kwargs)

        assert model.skip_copula is False, (
            "TACTiS2Model.skip_copula should be False"
        )
        assert model.tactis.skip_copula is False, (
            "TACTiS (inner model).skip_copula should be False when TACTiS2Model "
            "is created with skip_copula=False"
        )

    def test_skip_copula_propagates_to_decoder(self, _minimal_model_kwargs):
        """
        skip_copula must also reach the decoder inside the TACTiS instance.
        """
        model = TACTiS2Model(skip_copula=True, **_minimal_model_kwargs)

        assert hasattr(model.tactis, "decoder"), (
            "TACTiS should have a decoder attribute"
        )
        assert model.tactis.decoder.skip_copula is True, (
            "Decoder.skip_copula should be True when TACTiS2Model is created "
            "with skip_copula=True"
        )
