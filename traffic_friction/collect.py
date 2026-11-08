"""Measure traffic, vehicle dynamics, and outcomes from actual MetaDrive steps."""
from __future__ import annotations

import csv
import json
import logging
import math
import platform
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from . import __version__
from .controllers import SpeedGovernor, read_road_state
from .protocol import METHODS, Protocol, key, ACTIVE_TRAFFIC
from .rendering import capture_3d, topdown_frame

STEP_COLUMNS = [
    "episode_id", "map", "friction", "traffic_density", "seed", "controller",
    "step", "time_s", "x_m", "y_m", "speed_kmh", "target_speed_kmh",
    "lateral_error_m", "heading_error_rad", "curvature_inv_m", "risk",
    "steering", "throttle_brake", "traffic_vehicle_count",
    "nearest_vehicle_center_m", "vehicles_within_25m", "reward",
    "arrive_dest", "out_of_road", "crash_vehicle", "crash_object",
    "terminated", "truncated",
]
EPISODE_COLUMNS = [
    "episode_id", "map", "friction", "traffic_density", "seed", "controller",
    "friction_actual", "traffic_count_initial", "steps", "sim_time_s", "wall_time_s",
    "success", "out_of_road", "collision_vehicle", "collision_object", "timeout", "outcome",
    "mean_speed_kmh", "mean_abs_lateral_error_m", "p95_abs_lateral_error_m",
    "mean_abs_heading_error_rad", "braking_fraction", "mean_traffic_count",
    "traffic_exposure_fraction_25m", "min_vehicle_center_distance_m", "total_reward",
]


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def metadata_record(protocol: Protocol) -> dict:
    return dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        project_version=__version__, protocol=protocol.describe(),
        metadrive_version=package_version("metadrive-simulator"),
        python=sys.version, platform=platform.platform(),
        traffic_mode="respawn", random_traffic=False,
        visual_is_excluded_from_main_analysis=True,
        note="Observations from MetaDrive simulation, not physical road trials",
        controller_note="All speed governors share the same IDM steering and traffic response; no friction oracle",
        data_definition="Only completed episodes are committed to the CSV files",
    )


def make_env(map_name: str, friction: float, density: float, protocol: Protocol,
             render: bool = False):
    # Lazy imports let plotting and unit tests run without MetaDrive/Panda3D.
    from metadrive.envs.varying_dynamics_env import VaryingDynamicsEnv
    from metadrive.policy.idm_policy import IDMPolicy

    class SharedIDM(IDMPolicy):
        def __init__(self, control_object, random_seed):
            super().__init__(control_object, random_seed)
            self.enable_lane_change = False

    config = {
        "map": map_name,
        "num_scenarios": max(protocol.seeds) - min(protocol.seeds) + 1,
        "start_seed": min(protocol.seeds),
        "traffic_density": float(density),
        "traffic_mode": "respawn",
        "random_traffic": False,
        "need_inverse_traffic": False,
        "random_spawn_lane_index": False,
        "random_agent_model": False,
        "random_lane_width": False,
        "random_lane_num": False,
        "horizon": int(protocol.max_steps),
        "use_render": bool(render),
        "agent_policy": SharedIDM,
        "log_level": logging.ERROR,
        "random_dynamics": {
            "max_engine_force": None,
            "max_brake_force": None,
            "wheel_friction": (float(friction), float(friction)),
            "max_steering": None,
            "mass": None,
        },
    }
    return VaryingDynamicsEnv(config)


def traffic_observation(ego, traffic_manager):
    """Nearest center-to-center distance to a live non-ego traffic vehicle."""
    vehicles = traffic_manager.traffic_vehicles
    count = len(vehicles)
    if not count:
        return 0, float("nan"), 0
    ex, ey = map(float, ego.position[:2])
    distances = []
    for car in vehicles:
        if car is ego:
            continue
        x, y = map(float, car.position[:2])
        distances.append(math.hypot(x - ex, y - ey))
    if not distances:
        return count, float("nan"), 0
    return count, min(distances), sum(d <= 25.0 for d in distances)


def _write_csv_header(path: Path, columns: list[str]):
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf8") as file:
            csv.DictWriter(file, fieldnames=columns).writeheader()


def _existing_keys(file: Path) -> set[str]:
    if not file.exists():
        return set()
    with file.open(newline="", encoding="utf8") as stream:
        return {row["episode_id"] for row in csv.DictReader(stream)}


def _remove_orphan_steps(step_path: Path, completed: set[str]) -> int:
    """Roll back steps from a run interrupted between the two CSV commits."""
    tmp = step_path.with_suffix(".recovery.csv")
    dropped = 0
    with step_path.open("r", newline="", encoding="utf8") as source, \
         tmp.open("w", newline="", encoding="utf8") as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=STEP_COLUMNS)
        writer.writeheader()
        for row in reader:
            if row["episode_id"] in completed:
                writer.writerow(row)
            else:
                dropped += 1
    if dropped:
        tmp.replace(step_path)
        print(f"[RECOVERY] Removed {dropped} uncommitted step rows", flush=True)
    else:
        tmp.unlink(missing_ok=True)
    return dropped


def _safe_dynamics(env, friction: float) -> float:
    actual = float(env.agent.get_dynamics_parameters()["wheel_friction"])
    if not math.isclose(actual, friction, abs_tol=1e-5, rel_tol=0.0):
        raise RuntimeError(f"Actual wheel friction {actual} differs from requested {friction}")
    return actual


def _save_visual_frame(env, frames: list, visual_dir: Path, step: int, status: dict):
    if step >= 25 and not status["3d_attempted"]:
        status["3d_attempted"] = True
        try:
            capture_3d(env, visual_dir / "metadrive_3d.png")
            status["3d_saved"] = True
        except Exception as exc:
            status["3d_error"] = str(exc)
    if step % 6 == 0 and status["topdown_error"] is None:
        try:
            frame = topdown_frame(env)
            if step == 30:
                from PIL import Image
                Image.fromarray(frame).save(visual_dir / "metadrive_topdown.png")
            if step % 12 == 0 and len(frames) < 55:
                frames.append(frame)
        except Exception as exc:
            status["topdown_error"] = str(exc)


def run_episode(env, protocol: Protocol, map_name: str, friction: float, density: float,
                seed: int, controller: str, step_writer,
                visual_dir: Path | None = None) -> dict:
    env.reset(seed=int(seed))
    identifier = key(map_name, friction, density, seed, controller)
    policy = env.engine.get_policy(env.agent.name)
    if policy is None or not hasattr(policy, "target_speed"):
        raise RuntimeError("MetaDrive did not attach the expected IDMPolicy")
    actual_friction = _safe_dynamics(env, friction)
    manager = env.engine.traffic_manager
    initial_traffic = len(manager.traffic_vehicles)
    if density > 0.0 and initial_traffic < 1:
        raise RuntimeError(
            f"Requested density {density} but no traffic vehicles spawned on {map_name}; "
            "this condition cannot be reported as traffic exposure"
        )
    if density == 0.0 and initial_traffic:
        raise RuntimeError("Unexpected traffic in a zero-density control condition")

    governor = SpeedGovernor(controller, cruise_kmh=protocol.cruise_kmh)
    dt = float(env.config.get("physics_world_step_size", 0.02)) * float(env.config.get("decision_repeat", 5))
    measures = {k: [] for k in ("speed", "lateral", "heading", "braking", "traffic", "exposed", "distance")}
    total_reward = 0.0
    flags = dict(arrive_dest=False, out_of_road=False, crash_vehicle=False, crash_object=False)
    n_steps = 0
    wall_start = time.perf_counter()
    frames: list[np.ndarray] = []
    status = dict(**{"3d_attempted": False, "3d_saved": False, "3d_error": None,
                     "topdown_error": None, "gif_saved": False, "topdown_saved": False})
    if visual_dir is not None:
        visual_dir.mkdir(parents=True, exist_ok=True)

    for index in range(protocol.max_steps):
        prior = read_road_state(env.agent, policy)
        target, risk = governor.target(prior)
        policy.target_speed = float(target)
        _, reward, terminated, truncated, info = env.step([0.0, 0.0])
        n_steps = index + 1
        vehicle = env.agent
        post = read_road_state(vehicle, policy)
        traffic_count, nearest, near_count = traffic_observation(vehicle, manager)
        total_reward += float(reward)
        flags["arrive_dest"] |= bool(info.get("arrive_dest", False))
        flags["out_of_road"] |= bool(info.get("out_of_road", False))
        flags["crash_vehicle"] |= bool(info.get("crash_vehicle", False))
        flags["crash_object"] |= bool(info.get("crash_object", False))
        action = getattr(vehicle, "current_action", [float("nan"), float("nan")])
        row = dict(
            episode_id=identifier, map=map_name, friction=friction, traffic_density=density,
            seed=seed, controller=controller, step=n_steps, time_s=round(n_steps * dt, 5),
            x_m=float(vehicle.position[0]), y_m=float(vehicle.position[1]),
            speed_kmh=float(vehicle.speed_km_h), target_speed_kmh=float(target),
            lateral_error_m=float(post.lateral_error_m), heading_error_rad=float(post.heading_error_rad),
            curvature_inv_m=float(prior.curvature_abs_inv_m), risk=float(risk),
            steering=float(action[0]), throttle_brake=float(action[1]),
            traffic_vehicle_count=traffic_count, nearest_vehicle_center_m=nearest,
            vehicles_within_25m=near_count, reward=float(reward),
            arrive_dest=int(bool(info.get("arrive_dest", False))),
            out_of_road=int(bool(info.get("out_of_road", False))),
            crash_vehicle=int(bool(info.get("crash_vehicle", False))),
            crash_object=int(bool(info.get("crash_object", False))),
            terminated=int(bool(terminated)), truncated=int(bool(truncated)),
        )
        step_writer.writerow(row)
        measures["speed"].append(row["speed_kmh"])
        measures["lateral"].append(abs(row["lateral_error_m"]))
        measures["heading"].append(abs(row["heading_error_rad"]))
        measures["braking"].append(int(row["throttle_brake"] < -0.02))
        measures["traffic"].append(traffic_count)
        measures["exposed"].append(int(near_count > 0))
        if math.isfinite(nearest):
            measures["distance"].append(nearest)

        if visual_dir is not None:
            env.render(text={"Method": controller, "Friction": f"{friction:.2f}",
                             "Traffic": f"{density:.2f}", "Step": n_steps})
            _save_visual_frame(env, frames, visual_dir, n_steps, status)
        if terminated or truncated:
            break

    if not n_steps:
        raise RuntimeError("MetaDrive returned an episode with zero steps")

    if visual_dir is not None:
        from PIL import Image
        image_path = visual_dir / "metadrive_topdown.png"
        if frames and not image_path.exists():
            Image.fromarray(frames[0]).save(image_path)
        status["topdown_saved"] = image_path.is_file()
        if frames:
            images = [Image.fromarray(f) for f in frames]
            images[0].save(visual_dir / "metadrive_topdown.gif", save_all=True,
                           append_images=images[1:], duration=160, loop=0, optimize=True)
            status["gif_saved"] = True
        (visual_dir / "capture_status.json").write_text(json.dumps(status, indent=2), encoding="utf8")

    # Co-occurring flags are preserved. Exclusive outcomes prioritize collision.
    if flags["crash_vehicle"]:
        outcome = "vehicle_collision"
    elif flags["crash_object"]:
        outcome = "object_collision"
    elif flags["out_of_road"]:
        outcome = "out_of_road"
    elif flags["arrive_dest"]:
        outcome = "success"
    else:
        outcome = "timeout"
    return dict(
        episode_id=identifier, map=map_name, friction=friction, traffic_density=density,
        seed=seed, controller=controller, friction_actual=actual_friction,
        traffic_count_initial=initial_traffic, steps=n_steps, sim_time_s=n_steps * dt,
        wall_time_s=time.perf_counter() - wall_start,
        success=int(outcome == "success"), out_of_road=int(flags["out_of_road"]),
        collision_vehicle=int(flags["crash_vehicle"]), collision_object=int(flags["crash_object"]),
        timeout=int(outcome == "timeout"), outcome=outcome,
        mean_speed_kmh=float(np.mean(measures["speed"])),
        mean_abs_lateral_error_m=float(np.mean(measures["lateral"])),
        p95_abs_lateral_error_m=float(np.percentile(measures["lateral"], 95)),
        mean_abs_heading_error_rad=float(np.mean(measures["heading"])),
        braking_fraction=float(np.mean(measures["braking"])),
        mean_traffic_count=float(np.mean(measures["traffic"])),
        traffic_exposure_fraction_25m=float(np.mean(measures["exposed"])),
        min_vehicle_center_distance_m=float(min(measures["distance"])) if measures["distance"] else float("nan"),
        total_reward=total_reward,
    )


def _assert_resume_compatible(folder: Path, protocol: Protocol):
    recorded = json.loads((folder / "metadata.json").read_text(encoding="utf8"))
    normalized = json.loads(json.dumps(protocol.describe()))
    if recorded["protocol"] != normalized:
        raise RuntimeError("The requested profile does not match the existing run configuration")


def run_suite(folder: Path, protocol: Protocol) -> None:
    """Resume at episode boundaries; interrupted episode steps remain in tmp/ only."""
    folder.mkdir(parents=True, exist_ok=True)
    metadata_file = folder / "metadata.json"
    if metadata_file.exists():
        _assert_resume_compatible(folder, protocol)
    else:
        metadata_file.write_text(json.dumps(metadata_record(protocol), indent=2), encoding="utf8")
    episode_path = folder / "episodes.csv"
    step_path = folder / "steps.csv"
    _write_csv_header(episode_path, EPISODE_COLUMNS)
    _write_csv_header(step_path, STEP_COLUMNS)
    completed = _existing_keys(episode_path)
    with episode_path.open(encoding="utf8", newline="") as stream:
        count_saved = sum(1 for _ in csv.DictReader(stream))
    if len(completed) != count_saved:
        raise RuntimeError("Duplicate episode IDs in saved run. Do not resume until repaired")
    _remove_orphan_steps(step_path, completed)
    tmp_dir = folder / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    total = protocol.expected_episodes
    print(f"[SUITE] {protocol.name}: {total} expected episodes, {len(completed)} already recorded", flush=True)
    attempted = len(completed)
    for map_name in protocol.maps:
        for friction in protocol.frictions:
            for density in protocol.densities:
                tasks = [(seed, controller) for seed in protocol.seeds for controller in METHODS
                         if key(map_name, friction, density, seed, controller) not in completed]
                if not tasks:
                    continue
                env = make_env(map_name, friction, density, protocol)
                try:
                    for seed, controller in tasks:
                        episode_id = key(map_name, friction, density, seed, controller)
                        tmp = tmp_dir / "current_episode.csv"
                        with tmp.open("w", newline="", encoding="utf8") as scratch:
                            writer = csv.DictWriter(scratch, fieldnames=STEP_COLUMNS)
                            writer.writeheader()
                            result = run_episode(env, protocol, map_name, friction, density,
                                                 seed, controller, writer)
                        # Commit steps first and the episode record last. On an abnormal
                        # interruption after step append, the next resume detects it.
                        with step_path.open("a", newline="", encoding="utf8") as dest, \
                             tmp.open("r", newline="", encoding="utf8") as src:
                            target_writer = csv.DictWriter(dest, fieldnames=STEP_COLUMNS)
                            for row in csv.DictReader(src):
                                target_writer.writerow(row)
                        with episode_path.open("a", newline="", encoding="utf8") as dest:
                            writer = csv.DictWriter(dest, fieldnames=EPISODE_COLUMNS)
                            writer.writerow(result)
                        tmp.unlink(missing_ok=True)
                        completed.add(episode_id)
                        attempted += 1
                        print(f"[{attempted}/{total}] map={map_name} friction={friction:.2f} "
                              f"density={density:.2f} seed={seed} controller={controller:<10} "
                              f"outcome={result['outcome']:<18} live_traffic={result['traffic_count_initial']}", flush=True)
                finally:
                    env.close()
    print(f"[COLLECTION COMPLETE] {len(completed)}/{total} episodes: {folder}", flush=True)


def run_visual(folder: Path, profile: Protocol):
    """A separate demonstrative run, excluded from the scientific sample."""
    visual = folder / "visual"
    visual.mkdir(parents=True, exist_ok=True)
    for name in ("metadrive_3d.png", "metadrive_topdown.png", "metadrive_topdown.gif",
                 "capture_status.json", "steps.csv", "episodes.csv"):
        (visual / name).unlink(missing_ok=True)
    seed = max(profile.seeds) + 100
    protocol = Protocol("visual", (profile.maps[0],), (0.6,), (ACTIVE_TRAFFIC,), (seed,),
                        400, profile.cruise_kmh)
    env = make_env(protocol.maps[0], 0.6, ACTIVE_TRAFFIC, protocol, render=True)
    try:
        with (visual / "steps.csv").open("w", newline="", encoding="utf8") as file:
            writer = csv.DictWriter(file, fieldnames=STEP_COLUMNS)
            writer.writeheader()
            row = run_episode(env, protocol, protocol.maps[0], 0.6, ACTIVE_TRAFFIC,
                              seed, "adaptive", writer, visual_dir=visual)
        with (visual / "episodes.csv").open("w", newline="", encoding="utf8") as file:
            writer = csv.DictWriter(file, fieldnames=EPISODE_COLUMNS)
            writer.writeheader()
            writer.writerow(row)
        print(f"[VISUAL] {row['outcome']}: {visual}", flush=True)
    finally:
        env.close()
