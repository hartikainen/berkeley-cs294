import tensorflow as tf

MODEL_DIR_BASE = "./models"

def init_feature_columns(D_in):
    feature_columns = [
        tf.contrib.layers.real_valued_column(
            "",
            dimension=D_in,
            default_value=None,
            dtype=tf.float32,
            normalizer=None
        )
    ]

    return feature_columns

def create_baseline_model(D_in, D_out):
    feature_columns = init_feature_columns(D_in)

    model = tf.contrib.learn.DNNRegressor(
        model_dir=MODEL_DIR_BASE + "/baseline",
        feature_columns=feature_columns,
        hidden_units=[20],
        label_dimension=D_out,
        activation_fn=tf.nn.relu,
        dropout=0.0,
        optimizer=tf.train.AdamOptimizer(
          learning_rate=1e-2,
        )
    )

    return model

def create_multi_layer_model(D_in, D_out):
    feature_columns = init_feature_columns(D_in)

    model = tf.contrib.learn.DNNRegressor(
        model_dir=MODEL_DIR_BASE + "/multi-layer",
        feature_columns=feature_columns,
        hidden_units=[200, 400, 200],
        label_dimension=D_out,
        activation_fn=tf.nn.relu,
        dropout=0.5,
        optimizer=tf.train.AdamOptimizer(
          learning_rate=1e-2,
        )
    )
