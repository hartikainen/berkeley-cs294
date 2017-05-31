import pickle
import argparse
import numpy as np
import tensorflow as tf
import gym
from distutils.util import strtobool
from load_policy import load_policy

from helpers import train_test_val_split
AVAILABLE_ENVS = (
    'Ant-v1',
    'HalfCheetah-v1',
    'Hopper-v1',
    'Humanoid-v1',
    'Reacher-v1',
    'Walker2d-v1'
)

def get_expert_filename(env, num_rollouts):
    return "./expert_data/{}-{}.pkl".format(env, num_rollouts)

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
    parser.add_argument("--mode",
                        type=str,
                        choices=["train", "evaluate"],
                        help=("Mode to run in: train to train a model, "
                              "evaluate to evaluate ready trained model"))


    # parser.add_argument('--expert_file', type=str)
    # parser.add_argument('--data_file', type=str)

    args = vars(parser.parse_args())

    if args.get('expert_file', None) is None:
        args['expert_file'] = get_expert_filename(args['env'],
                                                  args['num_rollouts'])

    return args

def get_log_level(level):
    levels = {}
    levels[0] = logging.DEBUG
    levels[1] = logging.INFO
    levels[2] = logging.WARNING
    levels[3] = logging.ERROR
    levels[4] = logging.CRITICAL

    return levels[level]

def load_expert_data(expert_filename, verbose=False):
    """ Load the expert data from pickle saved in expert_filename"""

    expert_data = None
    with open(expert_filename, "rb") as f:
        expert_data = pickle.load(f)

    observations = expert_data["observations"].astype('float32')
    actions = np.squeeze(expert_data["actions"].astype('float32'))

    if verbose:
        # As a sanity check, print out the size of the training and test data.
        print('observations shape: ', observations.shape)
        print('actions shape: ', actions.shape)

    return observations, actions

def train_model(data):
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val =   data["X_val"],   data["y_val"]

    N_train = X_train.shape[0]
    D_out = y_train.shape[-1]

    BATCH_SIZE = 256
    NUM_EPOCHS = 2
    batches_per_epoch = int(N_train/BATCH_SIZE)

    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(
        X_train)
    model = tf.contrib.learn.DNNRegressor(
        model_dir="./models/checkpoints/initial-test-model",
        feature_columns=feature_columns,
        hidden_units=[100, 100, 100],
        label_dimension=D_out,
        activation_fn=tf.nn.relu,
        dropout=0.0,
        optimizer=tf.train.AdamOptimizer(
          learning_rate=1e-2,
        )
    )

    validation_metrics = {
        "rmse": tf.contrib.metrics.streaming_root_mean_squared_error,
        "accuracy": tf.contrib.metrics.streaming_accuracy,
    }

    early_stop_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        X_val,
        y_val,
        every_n_steps=50,
        metrics=validation_metrics,
        early_stopping_metric="rmse",
        early_stopping_metric_minimize=True,
        early_stopping_rounds=500)

    model.fit(
        x=X_train,
        y=y_train,
        monitors=[early_stop_monitor],
        steps=batches_per_epoch * NUM_EPOCHS
    )

def evaluate_model(data, model_dir, env, num_rollouts, expert_policy_file, render=False):
    X_val,  y_val =  data["X_val"],  data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(
        X_val)
    D_out = y_val.shape[-1]
    model = tf.contrib.learn.DNNRegressor(
        model_dir=model_dir,
        feature_columns=feature_columns,
        hidden_units=[100, 100, 100],
        label_dimension=D_out,
        activation_fn=tf.nn.relu,
        dropout=0.0,
    )

    env = gym.make(env)
    expert_policy = load_policy(expert_policy_file)

    returns = []

    max_steps = 100
    max_steps = env.spec.timestep_limit

    with tf.Session():
        for r in range(1, num_rollouts+1):
            obs = env.reset()
            done = False
            rollout_reward = 0.0
            steps = 0

            while not done and steps < max_steps:
                obs = np.array(obs)

                action = model.predict(x=obs[None, :], as_iterable=False)

                obs, reward, done, _ = env.step(action)

                rollout_reward += reward
                steps += 1

                if render: env.render()
                if steps % 100 == 0: print("%i/%i" % (steps, max_steps))

            returns.append(rollout_reward)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))



if __name__ == "__main__":
    args = parse_args()
    observations, actions = load_expert_data(args['expert_file'])
    train_prop = 16/20
    val_prop = 3/20
    test_prop = 1/20
    N_dev = 500
    data = train_test_val_split(
        observations, actions,
        train_prop, val_prop, test_prop, N_dev,
        verbose=True
    )

    if args['mode'] == "train":
        train_model(data)
    elif args['mode'] == "evaluate":
        pass
