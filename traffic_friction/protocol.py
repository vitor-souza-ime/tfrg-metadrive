"""Fixed experiment design and paired scenario identifiers.

The design includes exactly two traffic treatments: none and active traffic.
The simulator may quantize requested density values into the same fleet size;
we therefore do not describe 0.08 and 0.18 as distinct treatments.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from itertools import product

METHODS = ("fixed", "curvature", "adaptive")
NO_TRAFFIC = 0.0
ACTIVE_TRAFFIC = 0.18

@dataclass(frozen=True)
class Protocol:
    name: str
    maps: tuple[str, ...]
    frictions: tuple[float, ...]
    densities: tuple[float, ...]
    seeds: tuple[int, ...]
    max_steps: int
    cruise_kmh: float = 50.0

    def scenarios(self):
        return product(self.maps, self.frictions, self.densities, self.seeds)

    @property
    def expected_episodes(self) -> int:
        return (len(self.maps) * len(self.frictions) * len(self.densities)
                * len(self.seeds) * len(METHODS))

    def describe(self) -> dict:
        data = asdict(self)
        data.update(methods=list(METHODS), expected_episodes=self.expected_episodes)
        return data


PROTOCOLS = {
    "smoke": Protocol("smoke", ("CSC",), (0.3, 0.6),
                      (NO_TRAFFIC, ACTIVE_TRAFFIC), (8800,), 400),
    "preflight": Protocol("preflight", ("CSC", "SCS"), (0.6,),
                          (NO_TRAFFIC, ACTIVE_TRAFFIC), (8900, 8901), 550),
    "pilot": Protocol("pilot", ("CSC", "SCS"), (0.3, 0.45, 0.6),
                      (NO_TRAFFIC, ACTIVE_TRAFFIC), (9000, 9001, 9002), 650),
    "full": Protocol("full", ("CSC", "SCS"), (0.3, 0.4, 0.5, 0.6),
                     (NO_TRAFFIC, ACTIVE_TRAFFIC), tuple(s for s in range(9100, 9119) if s != 9105), 700),
}


def key(map_name: str, friction: float, density: float, seed: int, controller: str) -> str:
    return f"{map_name}|{friction:.3f}|{density:.3f}|{seed}|{controller}"
