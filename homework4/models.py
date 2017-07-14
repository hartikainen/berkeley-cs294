from functools import reduce
from operator import mul

import tensorflow as tf

class ModelConfig:
    D_in = (84, 84, 4)
    D_action = 1
    beta = 0.01
    stride_conv1 = 4
    N_kernels_conv1 = 16
    stride_conv2 = 2
    N_kernels_conv2 = 32
    D_fc1 = 256

DEFAULT_CONFIG = ModelConfig()

def conv_variables(shape):
    xavier_init = tf.contrib.layers.xavier_initializer()

    W = tf.Variable(xavier_init(shape))
    b = tf.Variable(tf.zeros((1, shape[-1])))

    return W, b

def fc_variables(shape):
    xavier_init = tf.contrib.layers.xavier_initializer()

    W_shape = (reduce(mul, shape[:-1], 1.0), shape[-1])
    W = tf.Variable(xavier_init(shape))
    b = tf.Variable(tf.zeros((1, shape[-1])))

    return W, b

def conv2d(x, W, stride, padding="VALID"):
    return tf.nn.conv2d(x, W, strides=(1, stride, stride, 1), padding=padding)


class ActionCriticValueFeedForward:
    def __init__(self, config=DEFAULT_CONFIG):
        """Initializes the model and builds the tensorflow graph for it.

        Args:
            config: A model configuration object of type Config
        """
        self.config = config
        self.build()

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        Adds following nodes to the computational graph

        observation_ph:
          Observation placeholder tensor of shape (None, D_observation), type tf.int32
        actions_ph:
          Actions placeholder tensor of shape (None, D_action), type tf.float32
        """
        D_in = self.config.D_in
        D_action = self.config.D_action

        self.observations_ph = tf.placeholder(
            tf.float32, (None, D_in[0], D_in[1], D_in[2]))
        self.actions_ph = tf.placeholder(tf.float32, (None, D_action))
        self.advantages_ph = tf.placeholder(tf.float32, [None])
        self.rewards_ph = tf.placeholder(tf.float32, [None])

    def init_variables(self):
        D_observation = self.config.D_in[2]
        D_action = self.config.D_action

        stride_conv1 = self.config.stride_conv1
        N_kernels_conv1 = self.config.N_kernels_conv1
        stride_conv2 = self.config.stride_conv2
        N_kernels_conv2 = self.config.N_kernels_conv2
        D_fc1 = self.config.D_fc1

        # Initialize variables
        # Convolution layer 1 (stride 4)
        self.W_conv1, self.b_conv1 = conv_variables((2*stride_conv1,
                                                     2*stride_conv1,
                                                     D_observation,
                                                     N_kernels_conv1))
        # Convolution layer 2 (stride 2)
        self.W_conv2, self.b_conv2 = conv_variables((2*stride_conv2,
                                                     2*stride_conv2,
                                                     N_kernels_conv1,
                                                     N_kernels_conv2))
        # Fully-connected layer # TODO: make 2592 variable
        self.W_fc1, self.b_fc1 = fc_variables([2592, D_fc1])
        # Policy layer
        self.W_pi, self.b_pi = fc_variables([D_fc1, D_action])
        # Value layer
        self.W_value, self.b_value = fc_variables([D_fc1, 1])


    def add_prediction_op(self):
        """Adds operators for a 3-hidden-layer neural network

        The network has the following architecture:
            conv - relu - conv - relu - affine - relu

        Returns:
            pred: tf.Tensor of shape (batch_size, n_classes)
        """
        observations = self.observations_ph

        stride_conv1 = self.config.stride_conv1
        stride_conv2 = self.config.stride_conv2

        z_conv1 = (conv2d(observations, self.W_conv1, stride_conv1)
                   + self.b_conv1)
        h_conv1 = tf.nn.relu(z_conv1)

        z_conv2 = conv2d(h_conv1, self.W_conv2, stride_conv2) + self.b_conv2
        h_conv2 = tf.nn.relu(z_conv2)
        # TODO make shape variable
        h_conv2_flat = tf.reshape(h_conv2, (-1, 2592))

        z_fc1 = tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1
        h_fc1 = tf.nn.relu(z_fc1)

        return h_fc1


    def add_loss_op(self, pred):
        # TODO: Check the signs!
        actions = self.actions_ph
        advantages = self.advantages_ph
        rewards = self.rewards_ph

        # Policy loss
        pi = tf.nn.softmax(tf.matmul(pred, self.W_pi)
                           + self.b_pi)
        pi = tf.clip_by_value(pi, 1e-20, 1.0)
        log_pi = tf.log(pi)

        action_probs = tf.reduce_sum(log_pi * actions, axis=1)
        policy_loss = - tf.reduce_sum(action_probs * advantages)

        # Value loss
        v = tf.matmul(pred, self.W_value) + self.b_value
        v_flat = tf.reshape(v, [-1])

        value_loss = tf.nn.l2_loss(rewards - v_flat)

        # Entropy loss
        # TODO: should this use tf.nn.softmax_cross_entropy_with_logits?
        entropy = - tf.reduce_sum(pi * log_pi, axis=1)

        self.loss = policy_loss + 0.5 * value_loss + self.config.beta * entropy


    def add_training_op(self, loss):
        pass


    def create_feed_dict(self, observations, actions=None):
        """Creates the feed_dict to be passed for tensorflow run function

        A feed_dict takes the form of:

        feed_dict = {
            <placeholder>: <tensor of values to be passed for placeholder>,
            ...
        }


        The keys for the feed_dict are a subset of the placeholder tensors
        created in add_placeholders. When an argument is None, we don't add
        it to the feed_dict.

        Args:
            observations: A batch of observation data.
            actions: A batch of action data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = { self.observations_ph: observations }

        if actions is not None:
            feed_dict.update({ self.actions_ph: actions })

        return feed_dict



    def build(self):
        self.add_placeholders()
        self.init_variables()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
