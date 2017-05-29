import pickle
import argparse
import numpy as np
from distutils.util import strtobool

AVAILABLE_ENVS = (
    'Ant-v1',
    'HalfCheetah-v1',
    'Hopper-v1',
    'Humanoid-v1',
    'Reacher-v1',
    'Walker2d-v1'
)

def parse_args():
    parser = argparse.ArgumentParser(description="Behavioral Cloning")

    parser.add_argument("--log",
                        type=int,
                        default=1,
                        help="Logging level")
    parser.add_argument("--env",
                        type=str,
                        default=AVAILABLE_ENVS[0],
                        choices=AVAILABLE_ENVS,
                        help="The name of the environment")
    parser.add_argument("--num_rollouts",
                        type=int,
                        default=20,
                        help="Number of expert roll outs")
    parser.add_argument("--render",
                        action="store_true",
                        help="Whether to render the MuJoCo environment")
    parser.add_argument("--max_timesteps",
                        type=int,
                        help=("Maximum number of steps to run environment for "
                              "each rollout"))

    # parser.add_argument('--expert_file', type=str)
    # parser.add_argument('--data_file', type=str)

    args = vars(parser.parse_args())
    return args

def get_log_level(level):
    levels = {}
    levels[0] = logging.DEBUG
    levels[1] = logging.INFO
    levels[2] = logging.WARNING
    levels[3] = logging.ERROR
    levels[4] = logging.CRITICAL

    return levels[level]

if __name__ == "__main__":
    args = parse_args()
    print(args)
