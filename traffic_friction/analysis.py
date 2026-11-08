"""Audit observed simulation records and create reproducible manuscript figures."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .collect import EPISODE_COLUMNS, STEP_COLUMNS
from .protocol import METHODS, PROTOCOLS, key

KEY_COLUMNS = ["map", "friction", "traffic_density", "seed"]
EXCLUSIVE_OUTCOMES = ["success", "out_of_road", "vehicle_collision", "object_collision", "timeout"]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path, required: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    missing = set(required) - set(table.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {sorted(missing)}")
    if table.empty:
        raise ValueError(f"No observations in {path.name}")
    return table


def verify_traffic_treatments(episodes: pd.DataFrame) -> dict:
    """Reject treatment labels that produced indistinguishable measured runs.

    The guard compares every matched (map, friction, seed, controller) pair.
    A difference in the requested density alone does not count as exposure.
    """
    levels = sorted(float(v) for v in episodes.traffic_density.unique())
    paired_keys = ["map", "friction", "seed", "controller"]
    measured = [
        "traffic_count_initial", "mean_traffic_count",
        "traffic_exposure_fraction_25m", "min_vehicle_center_distance_m",
        "steps", "success", "collision_vehicle", "out_of_road",
        "mean_speed_kmh", "mean_abs_lateral_error_m", "total_reward",
    ]
    report = {"requested_densities": levels, "comparisons": []}
    for i, left in enumerate(levels):
        a = episodes[episodes.traffic_density == left].set_index(paired_keys)
        for right in levels[i+1:]:
            b = episodes[episodes.traffic_density == right].set_index(paired_keys)
            common = a.index.intersection(b.index)
            if len(common) == 0:
                continue
            a = a.loc[common].sort_index()
            bb = b.loc[common].sort_index()
            same_rows = np.ones(len(common), dtype=bool)
            for field in measured:
                av = a[field].to_numpy(dtype=float)
                bv = bb[field].to_numpy(dtype=float)
                same_rows &= np.isclose(av, bv, rtol=1e-9, atol=1e-10, equal_nan=True)
            fraction_identical = float(same_rows.mean())
            different_active_fleet = bool(np.any(
                a["traffic_count_initial"].to_numpy() != bb["traffic_count_initial"].to_numpy()))
            report["comparisons"].append({
                "density_a": left, "density_b": right,
                "matched_conditions": len(common),
                "identical_measured_episodes": int(same_rows.sum()),
                "identical_fraction": fraction_identical,
                "different_initial_fleet_observed": different_active_fleet,
            })
            if same_rows.all():
                raise ValueError(
                    f"Duplicate traffic treatments: {left:.2f} and {right:.2f} "
                    f"produced identical measured runs in {len(common)} matched cases. "
                    "Change the design instead of treating them as independent conditions."
                )
    return report


def verify_exposure(episodes: pd.DataFrame) -> dict:
    """Confirm both maps have actual absence and presence of traffic."""
    rows = []
    for (road, density), group in episodes.groupby(["map", "traffic_density"]):
        counts = group.traffic_count_initial.to_numpy()
        proximity = group.traffic_exposure_fraction_25m.to_numpy()
        if density == 0 and (np.any(counts != 0) or np.any(proximity > 0)):
            raise ValueError(f"Unexpected traffic in no-traffic control on {road}")
        if density > 0 and (np.any(counts == 0) or not np.any(proximity > 0)):
            raise ValueError(
                f"Traffic treatment failed on {road}: expected live vehicles and actual "
                "ego proximity in at least one episode."
            )
        rows.append({
            "map": road, "requested_density": float(density),
            "n_episodes": len(group), "min_initial_vehicles": int(counts.min()),
            "mean_active_vehicles": float(group.mean_traffic_count.mean()),
            "mean_steps_with_vehicle_within_25m": float(proximity.mean()),
        })
    return {"verified_conditions": rows}

def audit(folder: Path, strict_complete: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Verify episode-step reconciliation and paired scenario representation."""
    meta_path = folder / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError("metadata.json is missing")
    metadata = json.loads(meta_path.read_text(encoding="utf8"))
    protocol = metadata["protocol"]
    e = _read(folder / "episodes.csv", EPISODE_COLUMNS)
    s = _read(folder / "steps.csv", STEP_COLUMNS)
    if e.episode_id.isna().any() or s.episode_id.isna().any():
        raise ValueError("Missing episode ID")
    if e.episode_id.duplicated().any():
        raise ValueError("Duplicate episodes detected")
    if s.duplicated(["episode_id", "step"]).any():
        raise ValueError("Duplicate step indices detected")
    if set(e.episode_id) != set(s.episode_id):
        raise ValueError("Episode IDs in steps.csv and episodes.csv do not agree")
    counts = s.groupby("episode_id").size()
    expected = e.set_index("episode_id")["steps"].astype(int)
    if not counts.sort_index().equals(expected.sort_index()):
        raise ValueError("Per-episode step counts disagree")
    first = s.groupby("episode_id")["step"].min()
    last = s.groupby("episode_id")["step"].max()
    if not ((first == 1).all() and (last.sort_index() == counts.sort_index()).all()):
        raise ValueError("Step indices must start at 1 and be contiguous")
    if (~e.controller.isin(METHODS)).any():
        raise ValueError("Unknown controller in results")
    if (~e.outcome.isin(EXCLUSIVE_OUTCOMES)).any():
        raise ValueError("Unknown terminal outcome")
    if not np.allclose(e["friction"], e["friction_actual"], atol=1e-5):
        raise ValueError("Measured friction differs from the requested friction")
    if (e.loc[e.traffic_density == 0, "traffic_count_initial"] != 0).any():
        raise ValueError("Unexpected traffic vehicles in zero-density scenarios")
    if (e.loc[e.traffic_density > 0, "traffic_count_initial"] < 1).any():
        raise ValueError("Nonzero traffic density produced no traffic vehicles")
    if (e.success != (e.outcome == "success").astype(int)).any():
        raise ValueError("Inconsistent success indicator")
    if (~e.traffic_density.isin(protocol["densities"])).any() or (~e.friction.isin(protocol["frictions"])).any():
        raise ValueError("CSV settings disagree with the experiment protocol")
    if (~e.seed.isin(protocol["seeds"])).any() or (~e["map"].isin(protocol["maps"])).any():
        raise ValueError("Unexpected seed or road geometry")
    expected_keys = {key(m, f, d, z, c) for m in protocol["maps"]
                     for f in protocol["frictions"] for d in protocol["densities"]
                     for z in protocol["seeds"] for c in METHODS}
    observed_keys = set(e.episode_id)
    if not observed_keys.issubset(expected_keys):
        raise ValueError("Extra episodes not defined by protocol")
    if strict_complete and observed_keys != expected_keys:
        raise ValueError(f"Incomplete experiment: {len(observed_keys)}/{len(expected_keys)} episodes")
    missing_values = e[["success", "mean_speed_kmh", "mean_abs_lateral_error_m", "mean_traffic_count"]].isna().sum().sum()
    if missing_values:
        raise ValueError("Non-finite primary episode measurements")
    # Check that every scenario has precisely the same 3 compared methods.
    if strict_complete:
        represented = e.groupby(KEY_COLUMNS).controller.agg(lambda x: set(x))
        if any(x != set(METHODS) for x in represented):
            raise ValueError("Unpaired scenario: missing controller")
    traffic_distinction = verify_traffic_treatments(e)
    exposure = verify_exposure(e)
    hashes = {name: _file_sha256(folder / name)
              for name in ("episodes.csv", "steps.csv", "metadata.json")}
    summary = {
        "episodes": int(len(e)), "recorded_steps": int(len(s)),
        "expected_episodes": len(expected_keys),
        "independent_seeds": int(e.seed.nunique()),
        "maps": sorted(e["map"].unique().tolist()),
        "frictions": sorted(e.friction.unique().tolist()),
        "traffic_densities": sorted(e.traffic_density.unique().tolist()),
        "outcomes": {k: int(v) for k, v in e.outcome.value_counts().to_dict().items()},
        "sha256": hashes,
        "traffic_treatment_validation": traffic_distinction,
        "exposure_validation": exposure,
        "note": "Data generated by MetaDrive; not physical-road observations",
    }
    (folder / "audit.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    return e, s, summary


def paired_differences(e: pd.DataFrame, n_boot: int = 1500) -> pd.DataFrame:
    """Paired differences; resample entire scenario seeds, not step records."""
    results = []
    n_seeds = e.seed.nunique()
    comparisons = (("adaptive", "fixed"), ("curvature", "fixed"),
                   ("adaptive", "curvature"))
    for density in ("all", *sorted(e.traffic_density.unique())):
        data = e if density == "all" else e[e.traffic_density == density]
        wide = data.pivot(index=KEY_COLUMNS, columns="controller", values="success")
        for left, right in comparisons:
            delta = (wide[left] - wide[right]).rename("diff")
            observed = float(delta.mean())
            lower = upper = float("nan")
            if n_seeds >= 5:
                per_seed = delta.groupby(level="seed").mean()
                rng = np.random.default_rng(2027)
                sampled = rng.choice(per_seed.to_numpy(), (n_boot, len(per_seed)), replace=True)
                lower, upper = map(float, np.quantile(sampled.mean(axis=1), [0.025, 0.975]))
                lower = min(lower, observed)
                upper = max(upper, observed)
            results.append(dict(density=density, method=left, reference=right,
                                delta=observed, ci95_low=lower, ci95_high=upper,
                                independent_seeds=n_seeds,
                                n_paired_conditions=int(len(delta))))
    return pd.DataFrame(results)


def traffic_interactions(e: pd.DataFrame, n_boot: int = 1500) -> pd.DataFrame:
    """Difference-in-differences in success for nonzero vs zero traffic.

    Compare each method with fixed-speed control in the SAME map, friction,
    and seed. Cluster bootstrap whole seeds; never individual steps.
    """
    densities = sorted(float(x) for x in e.traffic_density.unique())
    if 0.0 not in densities:
        raise ValueError("A zero-density reference condition is required")
    wide = e.pivot(index=["map", "friction", "seed"],
                   columns=["controller", "traffic_density"], values="success")
    rows = []
    n_seeds = e.seed.nunique()
    for density in densities:
        if density == 0:
            continue
        for left, right in (("adaptive", "fixed"), ("curvature", "fixed"),
                            ("adaptive", "curvature")):
            at_traffic = wide[(left, density)] - wide[(right, density)]
            at_zero = wide[(left, 0.0)] - wide[(right, 0.0)]
            diffs = (at_traffic - at_zero).rename("interaction")
            effect = float(diffs.mean())
            low = high = float("nan")
            if n_seeds >= 5:
                by_seed = diffs.groupby(level="seed").mean()
                rng = np.random.default_rng(808)
                bootstrap = rng.choice(by_seed.to_numpy(), (n_boot, len(by_seed)), replace=True).mean(axis=1)
                low, high = map(float, np.quantile(bootstrap, (0.025, 0.975)))
                low = min(low, effect)
                high = max(high, effect)
            rows.append(dict(traffic_density=density, method=left, reference=right,
                             difference_in_differences=effect,
                             ci95_low=low, ci95_high=high,
                             n_scenario_seeds=n_seeds, n_paired_groups=int(len(diffs))))
    return pd.DataFrame(rows)


def _save(fig, folder: Path, name: str):
    fig.tight_layout()
    fig.savefig(folder / f"{name}.png", dpi=210, bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _rate(data, axis, controller, xvar="friction", positive="success"):
    d = data[data.controller == controller].groupby(xvar)[positive].mean().sort_index()
    axis.plot(d.index, 100 * d.values, marker="o", label=controller.title())


def generate(folder: Path) -> dict:
    e, s, audit_info = audit(folder)
    tables = folder / "tables"
    figures = folder / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    overall = e.groupby("controller", as_index=False).agg(
        episodes=("success", "size"), success_rate=("success", "mean"),
        vehicle_collision_rate=("collision_vehicle", "mean"),
        out_of_road_rate=("out_of_road", "mean"),
        mean_speed_kmh=("mean_speed_kmh", "mean"),
        traffic_exposure=("traffic_exposure_fraction_25m", "mean"),
    )
    condition = e.groupby(["map", "friction", "traffic_density", "controller"], as_index=False).agg(
        episodes=("success", "size"), successes=("success", "sum"),
        success_rate=("success", "mean"), vehicle_collisions=("collision_vehicle", "sum"),
        vehicle_collision_rate=("collision_vehicle", "mean"),
        departures=("out_of_road", "sum"), out_of_road_rate=("out_of_road", "mean"),
        mean_speed_kmh=("mean_speed_kmh", "mean"),
        mean_traffic_count=("mean_traffic_count", "mean"),
        exposure_25m=("traffic_exposure_fraction_25m", "mean"),
    )
    outcomes = e.groupby(["traffic_density", "controller", "outcome"], as_index=False).size()
    success_times = e[e.success == 1].groupby(["traffic_density", "controller"], as_index=False).agg(
        successful_episodes=("success", "size"), mean_success_time_s=("sim_time_s", "mean")
    )
    paired = paired_differences(e)
    interaction = traffic_interactions(e)
    overall.to_csv(tables / "overall.csv", index=False)
    condition.to_csv(tables / "conditions.csv", index=False)
    outcomes.to_csv(tables / "outcomes.csv", index=False)
    success_times.to_csv(tables / "completion_times_success_only.csv", index=False)
    paired.to_csv(tables / "paired_effects.csv", index=False)
    interaction.to_csv(tables / "traffic_interactions.csv", index=False)
    pd.DataFrame(audit_info["exposure_validation"]["verified_conditions"]).to_csv(
        tables / "verified_traffic_exposure.csv", index=False)

    # 1. Safety response surface: average over paired seeds and maps.
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), sharex=True, sharey=True)
    for ax, name in zip(axes, METHODS):
        pivot = e[e.controller == name].pivot_table(
            index="friction", columns="traffic_density", values="success", aggfunc="mean"
        ).sort_index().sort_index(axis=1)
        im = ax.imshow(pivot.values * 100, origin="lower", vmin=0, vmax=100, aspect="auto")
        ax.set_title(name.title())
        ax.set_xticks(range(len(pivot.columns)), [f"{x:.2f}" for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [f"{x:.2f}" for x in pivot.index])
        ax.set_xlabel("Traffic density")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{100*pivot.iloc[i,j]:.0f}", ha="center", va="center", fontsize=8)
    axes[0].set_ylabel("Tire friction parameter")
    fig.colorbar(im, ax=axes.tolist(), fraction=0.03, pad=0.04, label="Completion (%)")
    fig.subplots_adjust(right=0.92, wspace=0.16)
    # Avoid tight_layout conflict with shared colorbar.
    fig.savefig(figures / "fig01_friction_traffic_surface.png", dpi=210, bbox_inches="tight")
    fig.savefig(figures / "fig01_friction_traffic_surface.pdf", bbox_inches="tight")
    plt.close(fig)

    # 2. Completion vs density, separately by map.
    fig, axes = plt.subplots(1, len(e["map"].unique()), figsize=(11, 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, map_name in zip(axes, sorted(e["map"].unique())):
        sample = e[e["map"] == map_name]
        for name in METHODS:
            _rate(sample, ax, name, "traffic_density", "success")
        ax.set(xlabel="Traffic density", title=f"Road layout: {map_name}", ylim=(-4, 104))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Completed episodes (%)")
    axes[-1].legend()
    _save(fig, figures, "fig02_completion_by_traffic_and_map")

    # 3. Explicit separation of collision and lane-departure mechanisms.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True)
    for ax, flag, title in zip(axes, ("collision_vehicle", "out_of_road"),
                               ("Vehicle collisions", "Departures from road")):
        for name in METHODS:
            _rate(e, ax, name, "traffic_density", flag)
        ax.set(xlabel="Traffic density", title=title, ylim=(0, 104))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Episodes with event (%)")
    axes[-1].legend()
    _save(fig, figures, "fig03_failure_modes_by_traffic")

    # 4. Conditional reliability against friction, one column for each density.
    densities = sorted(e.traffic_density.unique())
    fig, axes = plt.subplots(1, len(densities), figsize=(12.5, 4), sharey=True)
    for ax, density in zip(np.atleast_1d(axes), densities):
        sub = e[e.traffic_density == density]
        for name in METHODS:
            _rate(sub, ax, name)
        ax.set(xlabel="Tire friction parameter", title=f"Traffic density {density:.2f}", ylim=(-4, 104))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Completed episodes (%)")
    axes[-1].legend()
    _save(fig, figures, "fig04_completion_by_friction_and_traffic")

    # 5. Check actual manipulation: observed number of traffic vehicles.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    observed = e.groupby("traffic_density", as_index=False).agg(
        mean_vehicles=("mean_traffic_count", "mean"), exposure=("traffic_exposure_fraction_25m", "mean")
    )
    axes[0].plot(observed.traffic_density, observed.mean_vehicles, "o-")
    axes[0].set(xlabel="Requested density", ylabel="Mean active traffic vehicles", title="Traffic configuration check")
    axes[1].plot(observed.traffic_density, observed.exposure * 100, "o-")
    axes[1].set(xlabel="Requested density", ylabel="Steps with a vehicle within 25 m (%)", title="Observed ego exposure")
    for ax in axes:
        ax.grid(alpha=0.25)
    _save(fig, figures, "fig05_actual_traffic_exposure")

    # 6. Paired effect at each density, with intervals only for enough seeds.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, a, b in (("Adaptive - Fixed", "adaptive", "fixed"),
                        ("Adaptive - Curvature", "adaptive", "curvature")):
        d = paired[(paired.method == a) & (paired.reference == b) & (paired.density != "all")]
        x = d.density.astype(float).to_numpy()
        y = 100 * d.delta.to_numpy()
        if d.ci95_low.notna().all():
            lo = 100 * np.maximum(d.delta.to_numpy() - d.ci95_low.to_numpy(), 0)
            hi = 100 * np.maximum(d.ci95_high.to_numpy() - d.delta.to_numpy(), 0)
            ax.errorbar(x, y, yerr=np.stack([lo, hi]), marker="o", capsize=4, label=label)
        else:
            ax.plot(x, y, "o-", label=label)
    ax.axhline(0, linestyle="--", color="gray", linewidth=1)
    ax.set(xlabel="Traffic density", ylabel="Paired completion difference (p.p.)",
           title="Paired effects by traffic density")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, figures, "fig06_paired_completion_differences")

    # 7. Driving speed and completion; speed measured across ALL episodes.
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    by_method = e.groupby(["controller", "traffic_density"], as_index=False).agg(
        success=("success", "mean"), speed=("mean_speed_kmh", "mean")
    )
    for name in METHODS:
        part = by_method[by_method.controller == name].sort_values("traffic_density")
        ax.plot(part.speed, 100 * part.success, "o-", label=name.title())
        for r in part.itertuples():
            ax.annotate(f"{r.traffic_density:.2f}", (r.speed, 100*r.success), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel="Episode-mean speed (km/h)", ylabel="Completion (%)",
           title="Safety-efficiency trade-off (all episodes)")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, figures, "fig07_speed_completion_tradeoff")

    # 8. Matched real ego trajectories from a single scenario.
    selected = e[(e.traffic_density == max(densities)) & (e.friction == max(e.friction.unique()))]
    one = selected.sort_values(["map", "seed"]).iloc[0]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for name in METHODS:
        episode = key(one["map"], float(one.friction), float(one.traffic_density), int(one.seed), name)
        trajectory = s[s.episode_id == episode].sort_values("step")
        ax.plot(trajectory.x_m, trajectory.y_m, linewidth=1.6, label=name.title())
    ax.set(xlabel="Global x (m)", ylabel="Global y (m)", aspect="equal",
           title=f"Measured paths: {one['map']}, mu={one.friction:.2f}, density={one.traffic_density:.2f}, seed={one.seed}")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, figures, "fig08_matched_ego_trajectories")

    # 9. Successful-trip travel times only. Explicit denominator in title.
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for name in METHODS:
        sub = success_times[success_times.controller == name].sort_values("traffic_density")
        ax.plot(sub.traffic_density, sub.mean_success_time_s, "o-", label=name.title())
    ax.set(xlabel="Traffic density", ylabel="Mean successful-trip time (s)",
           title="Travel duration conditional on completion")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, figures, "fig09_successful_trip_duration")

    # 10. Traffic-by-controller interaction in completion probability.
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for label, left, right in (("Adaptive - Fixed", "adaptive", "fixed"),
                               ("Adaptive - Curvature", "adaptive", "curvature")):
        d = interaction[(interaction.method == left) & (interaction.reference == right)]
        x = d.traffic_density.to_numpy()
        y = 100 * d.difference_in_differences.to_numpy()
        if d.ci95_low.notna().all():
            lo = 100 * np.maximum(d.difference_in_differences.to_numpy() - d.ci95_low.to_numpy(), 0)
            hi = 100 * np.maximum(d.ci95_high.to_numpy() - d.difference_in_differences.to_numpy(), 0)
            ax.errorbar(x, y, yerr=np.stack([lo, hi]), capsize=4, marker="o", label=label)
        else:
            ax.plot(x, y, "o-", label=label)
    ax.axhline(0, linestyle="--", color="gray", linewidth=1)
    ax.set(xlabel="Traffic density", ylabel="Change in paired advantage vs zero traffic (p.p.)",
           title="Controller–traffic interaction in completion")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, figures, "fig10_traffic_controller_interaction")

    n_seeds = int(e.seed.nunique())
    report = (
        "# Traffic–friction experiment: recorded results\n\n"
        f"- Recorded completed episodes: {len(e):,}\n"
        f"- MetaDrive step observations: {len(s):,}\n"
        f"- Independent scenario seeds: {n_seeds}\n"
        f"- Map families: {', '.join(sorted(e['map'].unique()))}\n"
        f"- Traffic densities: {', '.join(format(x, '.2f') for x in densities)}\n"
        "- Vehicle collisions and lane departures are separate outcomes.\n"
        "- Traffic density is a simulator setting, not an observed vehicle flow in vehicles/hour.\n"
        "- Actual active vehicle counts and within-25-m exposure are measured each step.\n"
        "- Traffic positions are used for logging, not given to the speed governors.\n"
        "- Interval estimates resample whole scenario seeds, not time steps.\n"
        + ("- The small pilot is exploratory: no confidence intervals are reported.\n" if n_seeds < 5 else
           "- Bootstrap 95% intervals are supplied in tables/paired_effects.csv.\n")
        + "- Trip times are conditional on successful completion (selection bias is possible).\n"
        "- All reported data originate from local MetaDrive simulator execution, not physical experiments.\n"
        "- Do not infer superior methods without checking the paired effects and their uncertainty.\n"
    )
    (folder / "RESULTS.md").write_text(report, encoding="utf8")
    print(f"[ANALYSIS] {len(e)} episodes; {len(s)} steps; "
          f"{len(list(figures.glob('*.png')))} PNG charts; outputs in {folder}")
    return audit_info
