"""Friendly command-line entry point for baseline PySIMPLE cases."""

import argparse
import json
from pathlib import Path

from .api import run
from .cases import available_cases
from .performance import benchmark
from .postprocess import plot_result, save_result
from .solver import SimpleSolver
from .verification import developing_channel_metrics, poiseuille_metrics


def _parser():
    parser = argparse.ArgumentParser(description="Run a PySIMPLE Cartesian incompressible-flow case.")
    parser.add_argument("case_name", nargs="?", choices=available_cases(), help="case name (legacy positional form)")
    parser.add_argument("--case", choices=available_cases(), default="cavity", help="named baseline case (default: cavity)")
    parser.add_argument("--backend", default="numpy", choices=("numpy", "cupy"), help="array backend")
    parser.add_argument("--nx", type=int, help="physical cells in x")
    parser.add_argument("--ny", type=int, help="physical cells in y")
    parser.add_argument("--reynolds", type=float, default=100.0, help="cavity/channel Reynolds number")
    parser.add_argument("--lid-velocity", type=float, default=1.0, help="cavity lid or channel inlet velocity")
    parser.add_argument("--iterations", type=int, help="maximum SIMPLE iterations")
    parser.add_argument("--tolerance", type=float, help="continuity convergence tolerance")
    parser.add_argument("--pressure-solver", choices=("jacobi", "multigrid"), help="pressure-correction solver")
    parser.add_argument("--momentum-sweeps", type=int, help="Jacobi sweeps per momentum solve")
    parser.add_argument("--pressure-sweeps", type=int, help="Jacobi sweeps per pressure-correction solve")
    parser.add_argument("--output", type=Path, default=Path("result.npz"), help="solution archive path")
    parser.add_argument("--plot", type=Path, help="optional PNG plot path")
    parser.add_argument("--benchmark-repeats", type=int, default=0, help="run complete-solve timing repetitions")
    parser.add_argument("--benchmark-output", type=Path, help="optional JSON timing record")
    return parser


def main(argv=None):
    """Run one case and print concise convergence and verification information."""
    args = _parser().parse_args(argv)
    if args.case_name is not None and args.case != "cavity" and args.case != args.case_name:
        _parser().error("positional case and --case must agree")
    name = args.case_name or args.case
    completed = run(
        name, backend=args.backend, nx=args.nx, ny=args.ny, reynolds=args.reynolds,
        lid_velocity=args.lid_velocity, iterations=args.iterations, tolerance=args.tolerance,
        pressure_solver=args.pressure_solver, momentum_sweeps=args.momentum_sweeps,
        pressure_sweeps=args.pressure_sweeps,
    )
    save_result(args.output, completed.result, completed.case)
    print("case: {}".format(completed.case.name))
    print("grid: {} x {} physical cells".format(completed.case.grid.nx, completed.case.grid.ny))
    print("iterations: {}".format(completed.result.iterations))
    print("converged: {}".format(completed.result.converged))
    print("continuity residual: {:.6e}".format(completed.result.continuity_history[-1]))
    if name == "channel":
        print("channel metrics:", developing_channel_metrics(completed.result, completed.grid, completed.case))
    elif name == "poiseuille":
        print("verification:", poiseuille_metrics(completed.result, completed.grid, completed.case))
    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot_result(args.plot, completed.result, completed.grid)
    if args.benchmark_repeats:
        report = benchmark(completed.case, backend=args.backend, repeats=args.benchmark_repeats)
        print("benchmark:", report)
        if args.benchmark_output:
            args.benchmark_output.parent.mkdir(parents=True, exist_ok=True)
            args.benchmark_output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
