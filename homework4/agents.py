from helpers import discount

import tensorflow as tf
import gym


"""Holds agent hyperparams and meta information.

The config class is used to store various hyperparameters and meta
information parameters. Agent objects should be passed a Config() object at
instantiation.
"""
DEFAULT_AGENT_CONFIG = {
    "N_parallel_learners": 4,
    "n_iter": 100,
    "t_max": 10,
    "gamma": 0.98,
    "lr": 0.01,
    "min_lr": 0.002,
    "lr_decay_no_steps": 10000,
    "rnn_size": 10,
    "use_rnn": True,
    "num_threads": 1,
    "window_size": 4,
    "ob_shape": (84,84),
    "crop_centering": (0.5,0.7),
    "env_name": '',
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
        self.learner_config = learner_config
        self.model_config = model_config
        self.global_model = global_model
        self.stop_requested = False
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


    def learn(self, session):
        observations, actions, rewards, values = [], [], [], []
        env = self.env

        self.sync_params(session)

        timesteps_this_batch = 0
        for i in range(N_iterations):
            observation = env.reset()
            done, terminated = False, False
            episode_len = 0

            while not done:
                episode_len += 1

                policy, value_pred = self.local_model.run_pi_and_value(
                    session, observation)
                action = self._choose_action(policy)

                observations.append(observation)
                actions.append(action)
                value_preds.append(value_pred)

                observation, reward, done, _ = env.step(action)

                # TODO: clip rewards?
                rewards.append(reward)

            path = {
                "observations": np.array(observations),
                "value_preds": np.array(value_preds),
                "rewards": np.array(rewards),
                "actions": np.array(action)
            }

            paths.append(path)

            batch_timesteps += episode_len
            if bath_timesteps > min_timesteps_per_batch: break

        total_timesteps += batch_timesteps

        # Advantage function estimate
        value_targetss, value_predss, advantagess = [], [], []
        for path in paths:
            rewards, value_preds = path["rewards"], path["value_preds"]
            discounted_rewards = discount(rewards, self.config.gamma)
            advantages = discounted_rewards - value_preds

            value_targetss.append(discounted_rewards)
            value_predss.append(value_preds)
            advantagess.append(advantages)


        gradient_op = self.get_gradient_op()
        gradient_feed_dict = {
            self.local_model.observations_ph: batch_observations,
            self.local_model.actions_ph: batch_actions,
            self.local_model.advantages_ph: batch_advantages,
            self.local_model.rewards_ph: rewards,
            # TODO: learning rate placeholder
        }

        sess.run(gradient_op, feed_dict=gradient_feed_dict)


# TODO: this class should probably be a more generic AsynchronousAgent, that
# can use n-step Q-learning etc.
class A3CAgent:
    def __init__(self, env_name, ModelCls, config, model_config):
        self.env_name = env_name
        self.config = config
        self.global_model = ModelCls(model_config)


        learners = []
        for i in range(self.config["N_parallel_learners"]):
            learner_config = config.copy()
            learner_config["device"] = "/job:worker/task:{}/cpu:0".format(i)
            env = gym.make(env_name)
            learner = A3CThread(env,
                                ModelCls,
                                global_model,
                                learner_config,
                                model_config)
            learners.append(learner)

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                                allow_soft_placement=True))

        init = tf.global_variables_initializer()
        sess.run(init)
