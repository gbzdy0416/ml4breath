import tensorflow as tf

def custom_loss(y_true, y_pred):
    """
    自定义损失函数：基于 MSE，并在以下情况下增加惩罚：
      - 预测的 satisfaction < 2.75
      - 预测的 reduction >= 0.0

    假设输出格式为：[inhale, exhale, repetition, reduction, satisfaction, age_group, gender]
    reduction 是第 4 个（索引 3），satisfaction 是第 5 个（索引 4）
    """
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    pred_inhale = y_pred[:, 0]
    pred_exhale = y_pred[:, 1]
    pred_reduction = y_pred[:, 3]
    pred_satisfaction = y_pred[:, 4]

    penalty_satisfaction = tf.cast(pred_satisfaction < 2.75, tf.float32)
    penalty_reduction = tf.cast(pred_reduction >= 0.0, tf.float32)

    penalty = 0.1 * penalty_satisfaction + 0.2 * penalty_reduction + 0.05 * (pred_inhale - pred_exhale)

    return mse + penalty
