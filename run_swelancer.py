from __future__ import annotations

# Load environment before importing anything else
from dotenv import load_dotenv

load_dotenv()

import argparse

import nanoeval
from nanoeval.evaluation import EvalSpec, RunnerArgs
from nanoeval.recorder import dummy_recorder
from nanoeval.setup import nanoeval_entrypoint
from swelancer import SWELancerEval
from swelancer_agent import SimpleAgentSolver


def parse_args():
    parser = argparse.ArgumentParser(description="Run SWELancer evaluation")
    parser.add_argument(
        "--issue_ids",
        nargs="*",
        type=str,
        help="List of ISSUE_IDs to evaluate. If not specified, all issues will be evaluated.",
    )
    return parser.parse_args()


async def run_single_eval(issue_id):
    report = await nanoeval.run(
        EvalSpec(
            eval=SWELancerEval(
                solver=SimpleAgentSolver(model="gpt-4o"), taskset=[issue_id]
            ),
            runner=RunnerArgs(
                concurrency=1,  # Set to 1 to ensure sequential processing
                experimental_use_multiprocessing=False,  # Disable multiprocessing
                enable_slackbot=False,
                recorder=dummy_recorder(),
                max_retries=5,
            ),
        )
    )
    return report


async def main() -> None:
    args = parse_args()

    if args.issue_ids:
        # Run sequential evaluations for each issue_id
        all_reports = []
        for issue_id in args.issue_ids:
            print(f"Evaluating issue_id: {issue_id}")
            report = await run_single_eval(issue_id)
            all_reports.append((issue_id, report))
            print(f"Completed evaluation for issue_id: {issue_id}")
            print(report)
            print("-" * 50)

        # Print summary of all reports
        print("\nSummary of all evaluations:")
        for issue_id, report in all_reports:
            print(f"Issue ID: {issue_id}")
            print(report)
            print("-" * 30)
    else:
        # Run a single evaluation for all issues
        report = await nanoeval.run(
            EvalSpec(
                eval=SWELancerEval(
                    solver=SimpleAgentSolver(model="gpt-4o"), taskset=None
                ),
                runner=RunnerArgs(
                    concurrency=1,  # Set to 1 to ensure sequential processing
                    experimental_use_multiprocessing=False,  # Disable multiprocessing
                    enable_slackbot=False,
                    recorder=dummy_recorder(),
                    max_retries=5,
                ),
            )
        )
        print(report)


if __name__ == "__main__":
    nanoeval_entrypoint(main())
