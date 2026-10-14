import argparse
import re

from reproducibility.src.analysis_data import (
    DataConverter,
    DataGenerator,
    PackProcessDataGenerator,
)
from tqdm import tqdm



def df_analysis_per_env(seeds,env,path):

    # STEP 1: Collect data for the graphs by seed.
    pack = 'SB3_PPO_'+env
    costs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    optimal_train_n_ep, optimal_test_cost, optimal_test_n_ep = 50, 0.2, 5
    default_train_n_ep, default_test_n_ep = 100, 5
    all_default_test_freq={ # Environments included in the empirical analysis of the paper.
                            'LunarLanderContinuous':10,'BipedalWalker': 5,
                            'Ant': 5,'HalfCheetah': 20,'Hopper': 20,'Swimmer': 10,'Walker2d': 20,
                            # Additional environment as an example of reproducibility.
                            'Pendulum': 20}
    
    default_test_freq=all_default_test_freq[env]

    for seed in seeds:

        print(f"Starting seed {seed}")

        datagenerator = DataGenerator(pack, [seed],path)
        datagenerator.add_all_criteria_eff(optimal_train_n_ep,optimal_test_cost,optimal_test_n_ep,single_seed=seed)

        for cost in tqdm(costs):
            datagenerator.add_test_cost_truth(cost,default_test_n_ep,single_seed=seed)
            datagenerator.add_all_criteria_estimates(default_train_n_ep,default_test_n_ep,default_test_freq,optimal_train_n_ep,cost,optimal_test_n_ep,single_seed=seed)

        print(f"Analysis data generated for seed {seed}")

    # STEP 2: Joint the data from all seeds.
    DataConverter.join_df_analysis_seed(path+'/SB3_PPO_'+env+'/analysis_data/',seeds,only_test_with_cost_truth=False)
    DataConverter.join_df_analysis_seed(path+'/SB3_PPO_'+env+'/analysis_data/',seeds,only_test_with_cost_truth=True)
    DataConverter.join_df_analysis_seed(path+'/SB3_PPO_'+env+'/analysis_data/',seeds,only_default_est=True)

    # STEP 3: Additional data that does not need to be joined (it is already generated as a single set).
    PackProcessDataGenerator.df_analysis_all_seq_est(seeds,env,path)
    PackProcessDataGenerator.df_analysis_all_seq_val_cost_len(seeds,env,path)


def parse_seeds(values):
    """Parse either a list of integer seeds or a list(range(start, stop)) expression."""
    if len(values) == 1:
        match = re.fullmatch(
            r"list\(range\(\s*(-?\d+)\s*,\s*(-?\d+)(?:\s*,\s*(-?\d+))?\s*\)\)",
            values[0],
        )
        if match:
            start, stop, step = match.groups()
            return list(range(int(start), int(stop), int(step) if step else 1))
    try:
        return [int(value) for value in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Provide integer seeds (for example: --seeds 1 2 3) or a quoted "
            "list(range(start, stop)) expression."
        ) from exc


# Command-line entry point for generating analysis data for one environment.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and combine analysis data for one PPO environment."
    )
    parser.add_argument(
        "--seeds",
        required=True,
        nargs="+",
        help="Seeds to process, or a quoted expression such as 'list(range(1, 31))'.",
    )
    parser.add_argument(
        "--env",
        required=True,
        choices=[
            "LunarLanderContinuous", "BipedalWalker", "Walker2d", "Ant",
            "HalfCheetah", "Hopper", "Swimmer", "Pendulum",
        ],
        help="Environment name; Pendulum is an additional example outside the paper analysis.",
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Root directory containing the process data, e.g. reproducibility/results/data.",
    )
    args = parser.parse_args()
    df_analysis_per_env(parse_seeds(args.seeds), args.env, args.path)
