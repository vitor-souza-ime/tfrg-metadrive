"""Exercise the CSV collection interface with an in-process mock environment.

These unit-test objects do not run vehicle physics. They are not experimental
results and never appear in the distributed results directory.
"""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from traffic_friction.collect import run_episode, STEP_COLUMNS, traffic_observation
from traffic_friction.protocol import Protocol


class Lane:
    length = 100.0
    width = 3.5

    def local_coordinates(self, position):
        return position[0], position[1]

    def heading_theta_at(self, s):
        return 0.01 * s


class Vehicle:
    def __init__(self):
        self.name = "ego"
        self.position = [0.0, 0.2]
        self.lane = Lane()
        self.heading_theta = 0.0
        self.speed_km_h = 35.0
        self.current_action = [0.04, 0.1]

    def get_dynamics_parameters(self):
        return {"wheel_friction": 0.6}


class Engine:
    def __init__(self, density):
        self.policy = SimpleNamespace(target_speed=35.0)
        traffic = [] if density == 0.0 else [SimpleNamespace(position=[15., 0.0])]
        self.traffic_manager = SimpleNamespace(traffic_vehicles=traffic)

    def get_policy(self, name):
        return self.policy


class Environment:
    config = {"physics_world_step_size": 0.02, "decision_repeat": 5}

    def __init__(self, density):
        self.agent = Vehicle()
        self.engine = Engine(density)
        self.steps = 0

    def reset(self, seed):
        self.steps = 0
        self.agent.position = [0.0, 0.2]
        return None, {}

    def step(self, action):
        self.steps += 1
        self.agent.position[0] += 2
        done = self.steps == 3
        return None, 1.0, done, False, {"arrive_dest": done,
                                            "out_of_road": False,
                                            "crash_vehicle": False,
                                            "crash_object": False}


def test_traffic_observation_and_episode_logging():
    protocol = Protocol("fixture", ("CSC",), (0.6,), (0.18,), (50,), 10)
    with TemporaryDirectory() as d:
        target = Path(d) / "steps.csv"
        with target.open("w", newline="", encoding="utf8") as stream:
            writer = csv.DictWriter(stream, fieldnames=STEP_COLUMNS)
            writer.writeheader()
            outcome = run_episode(Environment(.18), protocol, "CSC", .6, .18, 50,
                                  "adaptive", writer)
        lines = list(csv.DictReader(target.open()))
        assert len(lines) == 3
        assert outcome["outcome"] == "success"
        assert outcome["steps"] == 3
        assert outcome["traffic_count_initial"] == 1
        assert outcome["traffic_exposure_fraction_25m"] == 1
        assert outcome["min_vehicle_center_distance_m"] > 0
        assert all(int(line["traffic_vehicle_count"]) == 1 for line in lines)


def test_complete_collection_analysis_and_resume(monkeypatch):
    """Complete pipeline test, using a fake environment only in a temp directory."""
    from traffic_friction import collect
    from traffic_friction.analysis import audit, generate
    from traffic_friction.protocol import PROTOCOLS

    class ClosedEnvironment(Environment):
        def close(self):
            pass

    def factory(map_name, friction, density, protocol, render=False):
        instance = ClosedEnvironment(density)
        instance.agent.get_dynamics_parameters = lambda: {"wheel_friction": friction}
        return instance

    monkeypatch.setattr(collect, "make_env", factory)
    with TemporaryDirectory() as d:
        output = Path(d) / "result"
        collect.run_suite(output, PROTOCOLS["smoke"])
        e, s, _ = audit(output)
        assert len(e) == 12 and len(s) == 36
        collect.run_suite(output, PROTOCOLS["smoke"])
        e2, s2, _ = audit(output)
        assert len(e2) == 12 and len(s2) == 36
        generate(output)
        assert (output / "figures" / "fig10_traffic_controller_interaction.pdf").is_file()
