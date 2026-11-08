"""Save frames from MetaDrive itself. Failures are reported, never substituted."""
from __future__ import annotations
import inspect
import re
from pathlib import Path
import numpy as np


def topdown_frame(env):
    from metadrive.engine.top_down_renderer import TopDownRenderer
    options = dict(window=False, screen_size=(900, 650), scaling=4,
                   draw_target_vehicle_trajectory=True, draw_center_line=True)
    try:
        signature = inspect.signature(TopDownRenderer.__init__)
        allowed = {n for n, p in signature.parameters.items() if p.kind in (
            inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)}
        if allowed:
            options = {k: v for k, v in options.items() if k in allowed}
    except (TypeError, ValueError):
        pass
    while True:
        try:
            frame = env.render(mode="topdown", **options)
            data = np.asarray(frame)
            if data.ndim != 3 or data.shape[-1] < 3:
                raise RuntimeError(f"Unexpected top-down frame dimensions: {data.shape}")
            return np.uint8(np.clip(data[:, :, :3], 0, 255))
        except TypeError as exc:
            rejected = re.search(r"unexpected keyword argument ['\"](\w+)['\"]", str(exc))
            if rejected is None or rejected.group(1) not in options:
                raise
            options.pop(rejected.group(1))


def capture_3d(env, output: Path):
    from panda3d.core import Filename
    window = getattr(env.engine, "win", None)
    if window is None:
        raise RuntimeError("No MetaDrive/Panda3D graphics window")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.saveScreenshot(Filename.fromOsSpecific(str(output.resolve()))):
        raise RuntimeError("Panda3D could not save the screenshot")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("3D screenshot was not created")
