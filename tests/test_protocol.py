from traffic_friction.controllers import RoadState, SpeedGovernor
from traffic_friction.protocol import PROTOCOLS, METHODS, key


def test_protocol_sizes_and_independent_seeds():
    assert PROTOCOLS["smoke"].expected_episodes == 12
    assert PROTOCOLS["preflight"].expected_episodes == 24
    assert PROTOCOLS["pilot"].expected_episodes == 108
    assert PROTOCOLS["full"].expected_episodes == 864
    for a, b in (("smoke", "pilot"), ("smoke", "full"), ("pilot", "full"),
                 ("smoke", "preflight"), ("preflight", "pilot"), ("preflight", "full")):
        assert not (set(PROTOCOLS[a].seeds) & set(PROTOCOLS[b].seeds))


def test_governors_share_identical_observation():
    state = RoadState(0.6, 0.05, 0.015, 3.5, 35.0)
    f = SpeedGovernor("fixed")
    c = SpeedGovernor("curvature")
    a = SpeedGovernor("adaptive")
    assert f.target(state) == (50.0, 0.0)
    curve, _ = c.target(state)
    adapt, risk = a.target(state)
    assert 9.0 <= adapt < curve <= 50.0
    assert risk > 0
    # The same state has no friction or traffic-density field.
    assert not hasattr(state, "friction")
    assert not hasattr(state, "traffic_density")


def test_pairing_key_is_deterministic():
    assert key("CSC", .3, .18, 7100, "fixed") == "CSC|0.300|0.180|7100|fixed"
