import tensorflow as tf
def custom_loss(y_true, y_pred):
    """
    custom loss (MSE +)：
      - When satisfaction < 3
      - When reduction >= 0.0
      - When inhale > exhale

    output: [inhale, exhale, repetition, reduction, satisfaction, age_group, gender]
    """
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    pred_inhale = y_pred[:, 0] * 1.784 + 5.075
    pred_exhale = y_pred[:, 1] * 1.683 + 5.289
    pred_reduction = y_pred[:, 3] * 38.1 + 50.0
    pred_satisfaction = y_pred[:, 4] * 0.75 + 3.25

    penalty_satisfaction = tf.nn.sigmoid(-(pred_satisfaction - 3.0) * 5.0)
    penalty_reduction = tf.nn.sigmoid(-1 * pred_reduction * 5.0)
    penalty_inhale = tf.nn.sigmoid((pred_inhale - pred_exhale) * 5.0)
    penalty = 0.1 * penalty_satisfaction + 0.1 * penalty_reduction + 0.2 * penalty_inhale

    return (mse + 0.5 * penalty)
