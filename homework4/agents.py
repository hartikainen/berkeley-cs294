import threading
import signal

import tensorflow as tf
import numpy as np
import gym

from helpers import discount


"""Holds agent hyperparams and meta information.

The config class is used to store various hyperparameters and meta
information parameters. Agent objects should be passed a Config() object at
instantiation.
"""
DEFAULT_AGENT_CONFIG = {
    "N_parallel_learners": 4,
    "local_t_max": 100,
    "global_t_max": 100,
    "gamma": 0.98,
    "learning_rate": 0.01,
    "lr_decay_no_steps": 10000,
    "rnn_size": 10,
    "use_rnn": False,
    "num_threads": 2,
    "window_size": 4,
    "ob_shape": (84,84),
    "crop_centering": (0.5,0.7),
    "env_name": 'CartPole-v0',
    "beta": 0.01,
    "rms_decay": 0.99,
    "rms_epsilon": 0.01,
    "random_starts": 30,
    "load_path": None,
    "train_dir": 'train',
    "evaluate": False,
    "render": False,
}

class A3CThread:
    def __init__(self, env, ModelCls, global_model, learner_config, model_config):
        self.env = env
        self.config = learner_config
        self.model_config = model_config
        self.global_model = global_model
        self.total_local_t = 0
        # actor critic and value function
        self.ModelCls = ModelCls
        self.local_model = ModelCls(model_config)

        self.build_gradient_ops()


    def build_gradient_ops(self):
        # TODO: finish this

        local_params = self.local_model.get_grad_params()
        global_params = self.global_model.get_grad_params()


        with tf.device(device):
            gradients = tf.gradients(
                self.local_model.loss,
                local_params,
                gate_gradients=False,
                aggregation_method=None,
                colocate_gradients_with_ops=False
            )


    def sync_params(self, session):
        global_params = self.global_model.get_sync_params()
        local_params  = self.local_model.get_sync_params()
        device_name = self.learner_config.device_name

        sync_ops = [
            tf.assign(local_param, global_param)
            for global_param, local_param in zip(global_params, local_params)
        ]

        session.run(sync_ops)


    def _choose_action(self, policy_probs):
        return np.random.choice(range(len(policy_probs)), p=policy_probs)


    def _get_gradient_op(self):
        pass


    def _anneal_learning_rate(self, global_t):
        initial_learning_rate = self.config["learning_rate"]
        global_t_max = self.config["global_t_max"]

        learning_rate = (initial_learning_rate
                         * (global_t_max - global_t)
                         / global_t_max)

        learning_rate = max(learning_rate, 0.0)

        return learning_rate


    def learn(self, session, global_t):
        observations, actions, rewards, value_preds = [], [], [], []
        env = self.env

        # TODO: reset gradients?
        self.sync_params(session)

        timesteps_this_batch = 0

        done, terminated = False, False
        observation = env.reset()

        local_t = 1
        while not done and local_t < self.config["local_t_max"]:
            policy, value_pred = self.local_model.run_policy_and_value(
                session, observation)
            action = self._choose_action(policy)

            observations.append(observation)
            actions.append(action)
            value_preds.append(value_pred)

            observation, reward, done, _ = env.step(action)

            # TODO: clip rewards?
            rewards.append(reward)

            local_t += 1

        if not done:
            bootstrap = self.local_network.run_value(sess, observations[-1])
        else:
            bootstrap = 0.0

        observations = np.array(observations)
        value_preds = np.array(value_preds)
        rewards = np.array(rewards)
        actions = np.array(action)

        # Advantage function estimate
        discounted_rewards = discount(rewards, self.config["gamma"], bootstrap)
        advantages = discounted_rewards - value_preds

        learning_rate = self._anneal_learning_rate(global_t)

        gradient_op = self._get_gradient_op()
        gradient_feed_dict = {
            self.local_model.observations_ph: observations,
            self.local_model.actions_ph: actions,
            self.local_model.advantages_ph: advantages,
            self.local_model.rewards_ph: discounted_rewards,
            self.learning_rate_ph: learning_rate
        }

        sess.run(gradient_op, feed_dict=gradient_feed_dict)

        self.total_local_t += local_t

        return local_t


# TODO: this class should probably be a more generic AsynchronousAgent, that
# can use n-step Q-learning etc.
class A3CAgent:
    def __init__(self, env_name, ModelCls, config, model_config):
        self.env_name = env_name
        self.env = env = gym.make(self.env_name)
        self.config = config
        self.model_config = model_config.copy()
        self.model_config.update({
            "D_observation": env.observation_space.shape[0],
            "D_action": env.action_space.n
        })
        self.ModelCls = ModelCls
        self.global_model = ModelCls(self.model_config)
        self.stop_requested = False


        self.init_learners()
        self.reset()


    def init_learners(self):
        learners = []
        for i in range(self.config["N_parallel_learners"]):
            learner_config = self.config.copy()
            device_name = "/job:worker/task:{}/cpu:0".format(i)
            learner_config["device_name"] = device_name
            env = self.env
            learner = A3CThread(env,
                                self.ModelCls,
                                self.global_model,
                                learner_config,
                                self.model_config)

            learners.append(learner)

        self.learners = learners


        init = tf.global_variables_initializer()
        session.run(init)


    def reset(self):
        self.global_t = 0


    def learn(self):
        global_t_max = self.config["global_t_max"]

        self.learners[0].learn(self.session, self.global_t)
        return

        def train_fn(learner):
            while self.global_t < global_t_max and not self.stop_requested:
                t = learner.learn(self.session, self.global_t)
                self.global_t += t


        def signal_handler(signal, frame):
            print('Stop Requested. Stopping learners.')
            self.stop_requested = True

        threads = [
            threading.Thread(target=train_fn, args=(self.learners[i],))
            for i in range(self.config["num_threads"])
        ]

        signal.signal(signal.SIGINT, signal_handler)

        for t in threads:
            t.start()

        print('Press Ctrl+C to stop')
        signal.pause()

        for t in threads:
            t.join()
