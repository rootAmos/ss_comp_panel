"""
MachLine example runner.
Runs three test cases (sphere, half-wing, compressible half-wing)
from MachLine's test suite and prints a summary of results.

Usage:
    cd /path/to/MachLine
    python /path/to/run_examples.py
"""

import json
import os
import subprocess
import sys


MACHLINE = os.path.join(os.path.dirname(__file__), "..", "MachLine") if \
    not os.path.exists("machline.exe") else "."
MACHLINE = os.path.abspath(MACHLINE)

EXAMPLES = [
    ("Sphere (incompressible)",          "test/input_files/sphere_input.json"),
    ("Half wing (incompressible)",       "test/input_files/half_wing_input.json"),
    ("Half wing (compressible M=0.5)",   "test/input_files/compressible_half_wing_input.json"),
    ("Full wing (incompressible)",       "test/input_files/full_wing_input.json"),
]


def run(input_file: str) -> dict:
    os.makedirs(os.path.join(MACHLINE, "test", "results"), exist_ok=True)
    result = subprocess.run(
        [os.path.join(MACHLINE, "machline.exe"), input_file],
        capture_output=True, text=True, cwd=MACHLINE,
    )
    report_path = os.path.join(MACHLINE, "test", "results", "report.json")
    if "MachLine exited successfully" not in result.stdout:
        print(result.stdout[-500:])
        print(result.stderr[-200:])
        return {}
    with open(report_path) as f:
        return json.load(f)


def fmt(v: float) -> str:
    return f"{v:+.5f}"


def main() -> None:
    print(f"MachLine directory: {MACHLINE}\n")
    print(f"{'Example':<40} {'Cx':>12} {'Cy':>12} {'Cz':>12}  {'time (s)':>10}")
    print("-" * 90)
    for label, inp in EXAMPLES:
        report = run(inp)
        if not report:
            print(f"{label:<40}  (FAILED)")
            continue
        forces = report.get("total_forces", {})
        t = report.get("total_runtime", 0.0)
        cx = fmt(forces.get("Cx", 0))
        cy = fmt(forces.get("Cy", 0))
        cz = fmt(forces.get("Cz", 0))
        print(f"{label:<40} {cx:>12} {cy:>12} {cz:>12}  {t:>10.3f}")
    print("\nAll examples completed.")


if __name__ == "__main__":
    sys.exit(main())
