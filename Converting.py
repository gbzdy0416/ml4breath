import coremltools as ct
import tensorflow as tf

# Loading Model
model=tf.keras.models.load_model('breath.h5', compile=False)

# Converting
mlmodel = ct.convert(
    model,
    source="tensorflow",
    inputs=[ct.TensorType(shape=(1, 3))],
    minimum_deployment_target=ct.target.iOS14
)

# Saving
mlmodel.save('BreathingModel.mlmodel')
