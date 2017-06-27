import argparse
from distutils.util import strtobool

import numpy as np
import tensorflow as tf
import gym

import models

AVAILABLE_ENVS = (
    'Ant-v1',
    'HalfCheetah-v1',
    'Hopper-v1',
    'Humanoid-v1',
    'Reacher-v1',
    'Walker2d-v1'
)

DEFAULT_ENV = 'Hopper-v1'

def parse_args():
    parser = argparse.ArgumentParser(
        description="DAgger - Dataset Aggregation")

    parser.add_argument("--env",
                        type=str,
                        default=DEFAULT_ENV,
                        choices=AVAILABLE_ENVS,
                        help="The name of the environment")
    parser.add_argument("--max_timesteps",
                        type=int,
                        default=5e5,
                        help="Number timesteps to run")
    parser.add_argument("--model_fn",
                        type=str,
                        default="default_model",
                        help=("Name of a function in models.py that will be "
                              "used as the model"))

    args = vars(parser.parse_args())

    return args


def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    available_gpus = [
        device_proto.physical_device_desc
        for device_proto in local_device_protos
        if device_proto.device_type == 'GPU'
    ]
    return available_gpus


def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1
    )
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session


def a3c_learn(env, model_fn, session, max_timesteps):
    pass


def main():
    args = parse_args()


    seed = 0
    env = gym.make(args['env'])
    # TODO: set seeds

    session = get_session()
    max_timesteps = args['max_timesteps'] or env.spec.timestep_limit

    model_fn = getattr(models, args['model_fn'])

    a3c_learn(env, model_fn, session, max_timesteps)


if __name__ == "__main__":
    main()
