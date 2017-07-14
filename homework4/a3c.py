from distutils.util import strtobool

import numpy as np
import tensorflow as tf
import gym

import models
import agents
from utils import parse_args

def discount(x, gamma):
    """
    Compute discounted sum of future values

    [https://github.com/berkeleydeeprlcourse/homework/blob/master/hw4/main.py]
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]


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
