"""Integration fixtures are synthetic and ephemeral; never shipped as results."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from traffic_friction.analysis import audit, generate, paired_differences
from traffic_friction.collect import EPISODE_COLUMNS, STEP_COLUMNS
from traffic_friction.protocol import PROTOCOLS, METHODS, key


def build_fixture(folder: Path, corrupt=False):
    protocol = PROTOCOLS["smoke"]
    (folder / "metadata.json").write_text(json.dumps({"protocol": protocol.describe()}), encoding="utf8")
    with (folder / "episodes.csv").open("w", newline="", encoding="utf8") as f1, \
         (folder / "steps.csv").open("w", newline="", encoding="utf8") as f2:
        episodes = csv.DictWriter(f1, fieldnames=EPISODE_COLUMNS)
        steps = csv.DictWriter(f2, fieldnames=STEP_COLUMNS)
        episodes.writeheader(); steps.writeheader()
        for road in protocol.maps:
            for friction in protocol.frictions:
                for density in protocol.densities:
                    for seed in protocol.seeds:
                        for method in METHODS:
                            identifier = key(road, friction, density, seed, method)
                            success = method != "fixed" or friction == .6
                            row = dict(episode_id=identifier, map=road, friction=friction,
                                       traffic_density=density, seed=seed, controller=method,
                                       friction_actual=friction, traffic_count_initial=3 if density > 0 else 0,
                                       steps=2, sim_time_s=.2, wall_time_s=.01,
                                       success=int(success), out_of_road=int(not success),
                                       collision_vehicle=0, collision_object=0, timeout=0,
                                       outcome="success" if success else "out_of_road",
                                       mean_speed_kmh=32., mean_abs_lateral_error_m=.2,
                                       p95_abs_lateral_error_m=.3,
                                       mean_abs_heading_error_rad=.1,
                                       braking_fraction=.2, mean_traffic_count=3 if density > 0 else 0,
                                       traffic_exposure_fraction_25m=.6 if density > 0 else 0,
                                       min_vehicle_center_distance_m=9.0 if density > 0 else float("nan"),
                                       total_reward=1.0)
                            episodes.writerow(row)
                            for step in (1, 2):
                                sr = {col: 0 for col in STEP_COLUMNS}
                                sr.update(episode_id=identifier, map=road, friction=friction,
                                          traffic_density=density, seed=seed, controller=method,
                                          step=step, time_s=step*.1, x_m=step*2., y_m=.2,
                                          speed_kmh=32., target_speed_kmh=40., lateral_error_m=.2,
                                          heading_error_rad=.03, curvature_inv_m=.01,
                                          risk=.1, steering=.1, throttle_brake=.1,
                                          traffic_vehicle_count=3 if density > 0 else 0,
                                          nearest_vehicle_center_m=9 if density > 0 else float("nan"),
                                          vehicles_within_25m=1 if density > 0 else 0,
                                          reward=.1, arrive_dest=int(success and step == 2),
                                          out_of_road=int(not success and step == 2),
                                          crash_vehicle=0, crash_object=0,
                                          terminated=int(step == 2), truncated=0)
                                steps.writerow(sr)
    if corrupt:
        with (folder / "steps.csv").open("a", encoding="utf8") as f:
            f.write("bad-id\n")


def test_audit_and_nine_graphs():
    with TemporaryDirectory() as d:
        folder = Path(d)
        build_fixture(folder)
        e, s, audit_info = audit(folder)
        assert len(e) == 12 and len(s) == 24
        result = generate(folder)
        assert len(list((folder / "figures").glob("*.png"))) == 10
        assert len(list((folder / "figures").glob("*.pdf"))) == 10
        assert (folder / "tables" / "paired_effects.csv").is_file()
        assert result["recorded_steps"] == 24


def test_mismatched_steps_fail_audit():
    with TemporaryDirectory() as d:
        folder = Path(d)
        build_fixture(folder, corrupt=True)
        with pytest.raises(ValueError):
            audit(folder)
