"""CLI for collecting and analyzing a reproducible MetaDrive traffic experiment."""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

from traffic_friction.protocol import PROTOCOLS, NO_TRAFFIC, ACTIVE_TRAFFIC

ROOT = Path(__file__).resolve().parent
PREFLIGHT_CERTIFICATE = ROOT / "results" / "PRECHECK_PASSED.json"


def archive(folder: Path) -> Path:
    destination = folder.parent / f"{folder.name}_SEND.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as output:
        for file in sorted(folder.rglob("*")):
            if file.is_file() and "tmp" not in file.relative_to(folder).parts:
                output.write(file, arcname=f"{folder.name}/{file.relative_to(folder).as_posix()}")
    with zipfile.ZipFile(destination) as check:
        bad = check.testzip()
        if bad is not None:
            raise RuntimeError(f"Archive integrity verification failed: {bad}")
    return destination


def new_run_directory(profile: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = ROOT / "results"
    folder = root / f"traffic_{profile}_{stamp}"
    index = 0
    while folder.exists():
        index += 1
        folder = root / f"traffic_{profile}_{stamp}_{index}"
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=list(PROTOCOLS), default="smoke")
    parser.add_argument("--folder", type=Path, help="Resume run in an existing run directory")
    parser.add_argument("--analyze", type=Path, help="Audit/plot existing CSV files; no simulation")
    parser.add_argument("--visual-only", type=Path, help="Recapture a 3D example without rerunning the benchmark")
    parser.add_argument("--skip-visual", action="store_true", help="Skip 3D/top-down captures")
    parser.add_argument("--plan", action="store_true", help="Show exact scenarios, do not run")
    args = parser.parse_args()
    profile = PROTOCOLS[args.profile]
    if args.plan:
        print(json.dumps(profile.describe(), indent=2))
        return
    from traffic_friction.analysis import generate
    if args.profile == "full" and not args.analyze and not args.visual_only and not args.plan:
        if not PREFLIGHT_CERTIFICATE.is_file():
            raise RuntimeError("Run RUN_PREFLIGHT.bat before the full experiment")
        checked = json.loads(PREFLIGHT_CERTIFICATE.read_text(encoding="utf8"))
        if not checked.get("passed") or checked.get("project_version") != "2.0.0":
            raise RuntimeError("Traffic precheck certificate is invalid or outdated")
    if args.visual_only:
        from traffic_friction.collect import run_visual
        folder = args.visual_only.resolve()
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf8"))
        recorded = metadata["protocol"]["name"]
        if recorded not in PROTOCOLS:
            raise ValueError(f"Unknown protocol in metadata: {recorded}")
        run_visual(folder, PROTOCOLS[recorded])
        destination = archive(folder)
        print("[SEND THIS ZIP]", destination)
        return
    if args.analyze:
        folder = args.analyze.resolve()
        generate(folder)
        destination = archive(folder)
        print("[SEND THIS ZIP]", destination)
        return
    from traffic_friction.collect import run_suite, run_visual
    folder = args.folder.resolve() if args.folder else new_run_directory(args.profile)
    folder.mkdir(parents=True, exist_ok=True)
    (ROOT / "LAST_RUN.txt").write_text(str(folder.resolve()), encoding="utf8")
    print("[RESULTS]", folder.resolve(), flush=True)
    run_suite(folder, profile)
    audit = generate(folder)
    if args.profile == "preflight":
        from traffic_friction import __version__
        levels = audit["traffic_treatment_validation"]["requested_densities"]
        if levels != [NO_TRAFFIC, ACTIVE_TRAFFIC]:
            raise RuntimeError("Preflight does not match the two required traffic treatments")
        status = {"passed": True, "project_version": __version__,
                  "source": str(folder.resolve()), "sha256": audit["sha256"],
                  "traffic_validation": audit["traffic_treatment_validation"],
                  "exposure": audit["exposure_validation"]}
        PREFLIGHT_CERTIFICATE.parent.mkdir(exist_ok=True)
        PREFLIGHT_CERTIFICATE.write_text(json.dumps(status, indent=2), encoding="utf8")
        print("[PREFLIGHT PASSED] Two actual traffic treatments confirmed on both maps")
    if args.profile == "full":
        import shutil
        shutil.copyfile(PREFLIGHT_CERTIFICATE, folder / "preflight_verification.json")
    if not args.skip_visual:

        visual_status = folder / "visual" / "capture_status.json"
        if not visual_status.exists():
            try:
                run_visual(folder, profile)
            except Exception as exc:
                (folder / "visual_error.txt").write_text(str(exc), encoding="utf8")
                print(f"[VISUAL WARNING] {exc} — benchmark CSVs are preserved", file=sys.stderr)
        if visual_status.exists():
            status = json.loads(visual_status.read_text(encoding="utf8"))
            if not status.get("3d_saved") or not status.get("topdown_saved"):
                print("[VISUAL WARNING] One or more captures failed; inspect capture_status.json")
    destination = archive(folder)
    print("\n[COLLECTION, ANALYSIS, PACKAGING COMPLETE]")
    print("[SEND THIS ZIP]", destination.resolve())


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\n[ERROR] {error}", file=sys.stderr)
        print("CSV files collected before the error remain in results/.\n"
              "Fix the issue and resume with: python main.py --profile PROFILE --folder RUN_DIRECTORY",
              file=sys.stderr)
        raise
