"""Reject traffic treatments that are different only in their labels."""

import pandas as pd
import pytest

from traffic_friction.analysis import verify_traffic_treatments, verify_exposure


def sample(density: float, seed: int) -> dict:
    active = density > 0
    return {
        "map": "CSC", "friction": 0.6, "seed": seed, "controller": "adaptive",
        "traffic_density": density, "traffic_count_initial": 3 if active else 0,
        "mean_traffic_count": 3.0 if active else 0.,
        "traffic_exposure_fraction_25m": 0.5 if active else 0.,
        "min_vehicle_center_distance_m": 8. if active else float("nan"),
        "steps": 300, "success": 1, "collision_vehicle": 0,
        "out_of_road": 0, "mean_speed_kmh": 30.,
        "mean_abs_lateral_error_m": 0.1, "total_reward": 12.,
    }


def test_observed_no_traffic_and_active_traffic_are_distinct():
    records = [sample(d, s) for d in (0., .18) for s in (8900, 8901)]
    result = verify_traffic_treatments(pd.DataFrame(records))
    assert result["comparisons"][0]["identical_fraction"] == 0.0
    assert verify_exposure(pd.DataFrame(records))["verified_conditions"]


def test_old_two_nonzero_levels_cannot_pass_as_different_traffic():
    records = [sample(d, s) for d in (.08, .18) for s in (8100, 8101)]
    with pytest.raises(ValueError, match="Duplicate traffic treatments"):
        verify_traffic_treatments(pd.DataFrame(records))


def test_missing_actual_proximity_fails_even_with_traffic_spawned():
    records = [sample(0., 8900), sample(.18, 8900)]
    records[1]["traffic_exposure_fraction_25m"] = 0.
    with pytest.raises(ValueError, match="Traffic treatment failed"):
        verify_exposure(pd.DataFrame(records))
