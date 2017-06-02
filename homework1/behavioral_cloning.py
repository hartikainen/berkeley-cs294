import pickle
import argparse
import numpy as np
import tensorflow as tf
import gym
from distutils.util import strtobool

import models
from helpers import train_test_val_split
from load_policy import load_policy


AVAILABLE_ENVS = (
    'Ant-v1',
    'HalfCheetah-v1',
    'Hopper-v1',
    'Humanoid-v1',
    'Reacher-v1',
    'Walker2d-v1'
)

LOG_LEVELS = (
    tf.logging.ERROR,
    tf.logging.INFO,
    tf.logging.WARN,
    tf.logging.ERROR,
    tf.logging.FATAL
)


def get_expert_data_file(env, num_rollouts):
    return "./expert_data/{}-{}.pkl".format(env, num_rollouts)


def get_expert_policy_file(env):
    return "./experts/{}.pkl".format(env)


def parse_args():
    parser = argparse.ArgumentParser(description="Behavioral Cloning")

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
    parser.add_argument("--mode",
                        type=str,
                        choices=["train", "evaluate"],
                        help=("Mode to run in: train to train a model, "
                              "evaluate to evaluate ready trained model"))
    parser.add_argument("--model_fn",
                        type=str,
                        default="create_baseline_model",
                        help=("Name of a function in models.py that returns a "
                              "model to be used for training/evaluation"))
    # parser.add_argument('--expert_file', type=str)

    args = vars(parser.parse_args())

    if args.get('expert_data_file', None) is None:
        args['expert_data_file'] = get_expert_data_file(args['env'],
                                                        args['num_rollouts'])

    if args.get('expert_policy_file', None) is None:
        args['expert_policy_file'] = get_expert_policy_file(args['env'])

    return args


def load_expert_data(filename, verbose=False):
    """ Load the expert data from pickle saved in filename"""

    expert_data = None
    with open(filename, "rb") as f:
        expert_data = pickle.load(f)

    observations = expert_data["observations"].astype('float32')
    actions = np.squeeze(expert_data["actions"].astype('float32'))

    if verbose:
        # As a sanity check, print out the size of the training and test data.
        print('observations shape: ', observations.shape)
        print('actions shape: ', actions.shape)

    return observations, actions


def init_monitors(X, y, every_n_steps=50, early_stopping_rounds=500):
    validation_metrics = {
        "rmse": tf.contrib.metrics.streaming_root_mean_squared_error,
        "accuracy": tf.contrib.metrics.streaming_accuracy,
    }

    early_stop_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        X, y,
        every_n_steps=every_n_steps,
        metrics=validation_metrics,
        early_stopping_metric="rmse",
        early_stopping_metric_minimize=True,
        early_stopping_rounds=early_stopping_rounds)

    return [early_stop_monitor]


def input_fn(X, y):
    feature_cols = { "observations": tf.constant(X, dtype=tf.float32) }
    labels = tf.constant(y)

    return feature_cols, labels


def train_model(model, data):
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val =   data["X_val"],   data["y_val"]

    N_train = X_train.shape[0]
    D_out = y_train.shape[-1]

    BATCH_SIZE = 256
    MAX_EPOCHS = 3
    batches_per_epoch = int(N_train/BATCH_SIZE)

    monitors = init_monitors(X_val, y_val)

    model.fit(
        # input_fn=lambda: input_fn(X_train, y_train),
        x=X_train,
        y=y_train,
        monitors=monitors,
        steps=batches_per_epoch * MAX_EPOCHS
    )


def evaluate_model(model, data, env, num_rollouts, expert_policy_file,
                   max_timesteps=None, render=False):
    env = gym.make(env)
    expert_policy = load_policy(expert_policy_file)

    returns = []

    if max_timesteps is None: max_timesteps = env.spec.timestep_limit

    with tf.Session():
        for r in range(1, num_rollouts+1):
            obs = env.reset()
            done = False
            rollout_reward = 0.0
            steps = 0

            while not done and steps < max_timesteps:
                obs = np.array(obs)

                action = model.predict(x=obs[None, :], as_iterable=False)

                obs, reward, done, _ = env.step(action)

                rollout_reward += reward
                steps += 1

                if render: env.render()
                if steps % 100 == 0: print("%i/%i" % (steps, max_timesteps))

            returns.append(rollout_reward)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))


if __name__ == "__main__":
    args = parse_args()

    tf.logging.set_verbosity(LOG_LEVELS[args['log_level']])

    observations, actions = load_expert_data(args['expert_data_file'])
    train_prop = 16/20
    val_prop = 3/20
    test_prop = 1/20
    N_dev = 500
    data = train_test_val_split(
        observations, actions,
        train_prop, val_prop, test_prop, N_dev,
        verbose=True
    )

    D_in, D_out = data["X_train"].shape[-1], data["y_train"].shape[-1]

    model_fn = getattr(models, args["model_fn"])
    model = model_fn(D_in, D_out)

    if args['mode'] == "train":
        train_model(model, data)
    elif args['mode'] == "evaluate":
        evaluate_model(model, data, env=args['env'],
                       num_rollouts=args['num_rollouts'],
                       expert_policy_file=args['expert_policy_file'])
