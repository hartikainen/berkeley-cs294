import argparse

import tensorflow as tf

AVAILABLE_ENVS = (
    "Atari40M",
)

LOG_LEVELS = (
    tf.logging.ERROR,
    tf.logging.INFO,
    tf.logging.WARN,
    tf.logging.ERROR,
    tf.logging.FATAL
)

def parse_args():
    parser = argparse.ArgumentParser(description="dqn")

    parser.add_argument("--log_level",
                        type=int,
                        default=4,
                        choices=tuple(range(len(LOG_LEVELS))),
                        help="Logging level")
    parser.add_argument("--env",
                        type=str,
                        default=AVAILABLE_ENVS[0],
                        choices=AVAILABLE_ENVS,
                        help="The name of the environment")
    parser.add_argument("--max_timesteps",
                        type=int,
                        help=("Maximum number of steps to run environment for "
                              "each rollout"))
    parser.add_argument("--mode",
                        type=str,
                        choices=["dev", "real"],
                        default="real",
                        help=("Mode to run in: 'dev' to test a model, "
                              "'real' for a real run"))
    parser.add_argument("--model_fn",
                        type=str,
                        default="atari_model",
                        help=("Name of a function in models.py that will be "
                              "used as the Q function"))
    parser.add_argument('--results_file',
                        type=str,
                        help="File path to write the results to")

    args = vars(parser.parse_args())

    return args
