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

class ActorLearnerThread:
    def __init__(self, env, ModelCls, learner_config, model_config):
        self.env = env
        self.learner_config = learner_config
        self.model_config = model_config
        self.ModelCls = ModelCls
        # actor critic and value function
        self.ac_vf = ModelCls(model_config)
        pass

class A3CAgent:
    def __init__(self, env_name, ModelCls, config, model_config):
        self.env_name = env_name
        self.config = config

        learners = []
        from pdb import set_trace; set_trace()
        for i in range(self.config["N_parallel_learners"]):
            learner_config = config.copy()
            learner_config["device"] = "/job:worker/task:{}/cpu:0".format(i)
            env = gym.make(env_name)
            learner = ActorLearnerThread(env, ModelCls, learner_config, model_config)
            learners.append(learner)
