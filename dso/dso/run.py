"""Parallelized, single-point launch script to run DSO on a set of benchmarks."""

import os
import sys
import time
from copy import deepcopy
from datetime import datetime

import click

from dso import DeepSymbolicOptimizer
from dso.logeval import LogEval
from dso.config import load_config
from dso.utils import safe_update_summary


def train_dso(config):
    """Trains DSO and returns dict of reward, expression, and traversal"""

    print("\n== TRAINING SEED {} START ============".format(config["experiment"]["seed"]))

    # Train the model
    model = DeepSymbolicOptimizer(deepcopy(config))
    start = time.time()
    result = model.train()
    result["t"] = time.time() - start
    result.pop("program")

    save_path = model.config_experiment["save_path"]
    summary_path = os.path.join(save_path, "summary.csv")

    print("== TRAINING SEED {} END ==============".format(config["experiment"]["seed"]))

    return result, summary_path


def print_summary(config, runs, messages):
    text = '\n== EXPERIMENT SETUP START ===========\n'
    text += 'Task type            : {}\n'.format(config["task"]["task_type"])
    if config["task"]["task_type"] == "regression":
        text += 'Dataset              : {}\n'.format(config["task"]["dataset"])
    text += 'Starting seed        : {}\n'.format(config["experiment"]["seed"])
    text += 'Runs                 : {}\n'.format(runs)
    if len(messages) > 0:
        text += 'Additional context   :\n'
        for message in messages:
            text += "      {}\n".format(message)
    text += '== EXPERIMENT SETUP END ============='
    print(text)


@click.command()
@click.argument('config_template', default="")
@click.option('--runs', '--r', default=1, type=int, help="Number of independent runs with different seeds")
@click.option('--seed', '--s', default=None, type=int, help="Starting seed (overwrites seed in config), incremented for each independent run")
@click.option('--benchmark', '--b', default=None, type=str, help="Name of benchmark")
@click.option('--exp_name', default=None, type=str, help="Name of experiment to manually generate log path")
def main(config_template, runs, seed, benchmark, exp_name):
    """Runs DSO in parallel across multiple seeds using multiprocessing."""

    messages = []

    # Load the experiment config
    config_template = config_template if config_template != "" else None
    config = load_config(config_template)

    # Overwrite named benchmark (for tasks that support them)
    task_type = config["task"]["task_type"]
    if benchmark is not None:
        # For regression, --b overwrites config["task"]["dataset"]
        if task_type == "regression":
            config["task"]["dataset"] = benchmark
        else:
            raise ValueError("--b is not supported for task {}.".format(task_type))

    # Update save dir if provided
    if exp_name is not None:
        config["experiment"]["exp_name"] = exp_name

    # Overwrite config seed, if specified
    if seed is not None:
        if config["experiment"]["seed"] is not None:
            messages.append(
                "INFO: Replacing config seed {} with command-line seed {}.".format(
                    config["experiment"]["seed"], seed))
        config["experiment"]["seed"] = seed

    # Save starting seed and run command
    config["experiment"]["starting_seed"] = config["experiment"]["seed"]
    config["experiment"]["cmd"] = " ".join(sys.argv)

    # Set timestamp once to be used by all workers
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    config["experiment"]["timestamp"] = timestamp

    # Start training
    print_summary(config, runs, messages)

    # Generate configs (with incremented seeds) for each run
    configs = [deepcopy(config) for _ in range(runs)]
    for i, config in enumerate(configs):
        config["experiment"]["seed"] += i

    # Farm out the work
    for i, config in enumerate(configs):
        result, summary_path = train_dso(config)
        if not safe_update_summary(summary_path, result):
            print("Warning: Could not update summary stats at {}".format(summary_path))
        print("INFO: Completed run {} of {} in {:.0f} s".format(i + 1, runs, result["t"]))

    # Evaluate the log files
    print("\n== POST-PROCESS START =================")
    log = LogEval(config_path=os.path.dirname(summary_path))
    log.analyze_log(
        show_count=config["postprocess"]["show_count"],
        show_hof=config["logging"]["hof"] is not None and config["logging"]["hof"] > 0,
        show_pf=config["logging"]["save_pareto_front"],
        save_plots=config["postprocess"]["save_plots"])
    print("== POST-PROCESS END ===================")


if __name__ == "__main__":
    main()
