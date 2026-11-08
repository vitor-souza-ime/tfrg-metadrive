"""Reference speed governors sharing an identical MetaDrive IDM steering policy.

The governor does not receive wheel friction, traffic density, or simulation
outcome flags. Traffic interactions are handled by the same IDMPolicy across
all three conditions. Equations reproduce the previously evaluated baselines.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def wrap_angle(radians: float) -> float:
    return math.atan2(math.sin(radians), math.cos(radians))


@dataclass(frozen=True)
class RoadState:
    lateral_error_m: float
    heading_error_rad: float
    curvature_abs_inv_m: float
    lane_width_m: float
    speed_kmh: float


def read_road_state(vehicle, policy=None, preview_m: float = 12.0) -> RoadState:
    """Measure curvature and tracking errors from the current simulation state."""
    lane = getattr(vehicle, "lane", None)
    if lane is None and policy is not None:
        lane = getattr(policy, "routing_target_lane", None)
    if lane is None:
        navigation = getattr(vehicle, "navigation", None)
        lanes = getattr(navigation, "current_ref_lanes", None)
        lane = lanes[0] if lanes else None
    if lane is None:
        raise RuntimeError("Vehicle lane localization is unavailable")

    longitudinal, lateral = lane.local_coordinates(vehicle.position)
    end = float(lane.length)
    s0 = clip(longitudinal, 0.0, end)
    s1 = clip(s0 + preview_m, 0.0, end)
    if s1 - s0 < 2.0:
        s0 = clip(s0 - preview_m, 0.0, end)
    h0 = float(lane.heading_theta_at(s0))
    h1 = float(lane.heading_theta_at(s1))
    curvature = abs(wrap_angle(h1 - h0)) / max(s1 - s0, 1.0)
    width = getattr(lane, "width", None)
    if width is None:
        width = vehicle.navigation.get_current_lane_width()
    return RoadState(float(lateral), wrap_angle(float(vehicle.heading_theta) - h0),
                     curvature, float(width), float(vehicle.speed_km_h))


@dataclass
class SpeedGovernor:
    name: str
    cruise_kmh: float = 50.0
    min_kmh: float = 9.0
    lateral_acc_limit_m_s2: float = 1.7
    risk_ema: float = 0.0

    def __post_init__(self):
        if self.name not in {"fixed", "curvature", "adaptive"}:
            raise ValueError(f"Unrecognized governor: {self.name}")

    def target(self, state: RoadState) -> tuple[float, float]:
        """Return target km/h and a risk state; friction is never an input."""
        if self.name == "fixed":
            return self.cruise_kmh, 0.0
        curve_limit = 3.6 * math.sqrt(
            self.lateral_acc_limit_m_s2 / max(state.curvature_abs_inv_m, 1e-6)
        )
        base = clip(curve_limit, self.min_kmh, self.cruise_kmh)
        if self.name == "curvature":
            return base, 0.0

        offset = abs(state.lateral_error_m) / max(0.5 * state.lane_width_m, 0.1)
        risk = clip(0.70 * offset + 1.10 * abs(state.heading_error_rad), 0.0, 1.0)
        self.risk_ema = 0.75 * self.risk_ema + 0.25 * risk
        reduction = clip(1.0 - 0.70 * self.risk_ema, 0.3, 1.0)
        return clip(base * reduction, self.min_kmh, self.cruise_kmh), self.risk_ema
