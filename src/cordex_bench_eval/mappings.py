"""
Shared mapping tables used by data loaders.
"""

from __future__ import annotations

# Training periods by experiment type
TRAIN_PERIODS = {
    "Emulator_hist_future": "1961-1980_2080-2099",
    "ESD_pseudo_reality": "1961-1980",
}

# Date tags for test periods
PERIOD_DATES = {
    "historical": "1981-2000",
    "mid_century": "2041-2060",
    "end_century": "2080-2099",
}

# GCM models used for training by domain
GCM_TRAIN = {
    "NZ": "ACCESS-CM2",
    "ALPS": "CNRM-CM5",
    "SA": "ACCESS-CM2",
}

# GCM models used for evaluation (hard-transferability)
GCM_EVAL = {
    "NZ": "EC-Earth3",
    "ALPS": "MPI-ESM-LR",
    "SA": "NorESM2-MM",
}

# Evaluation settings:
# training_experiment -> evaluation_experiment -> (period, condition, same_gcm_as_train)
EVAL_SETTINGS = {
    "ESD_pseudo_reality": {
        "Perfect_cross_validation": ("historical", "perfect", True),
        "Imperfect_cross_validation": ("historical", "imperfect", True),
        "Perfect_extrapolation": ("mid_end_century", "perfect", True),
        "Imperfect_extrapolation": ("mid_end_century", "imperfect", True),
        "Perfect_extrapolation_GCM_transferability": ("mid_end_century", "perfect", False),
    },
    "Emulator_hist_future": {
        "Perfect_cross_validation": ("historical", "perfect", True),
        "Imperfect_cross_validation": ("historical", "imperfect", True),
        "Perfect_interpolation": ("mid_century", "perfect", True),
        "Imperfect_interpolation": ("mid_century", "imperfect", True),
        "Perfect_interpolation_GCM_transferability": ("mid_century", "perfect", False),
        "Imperfect_interpolation_GCM_transferability": ("mid_century", "imperfect", False),
    },
}

