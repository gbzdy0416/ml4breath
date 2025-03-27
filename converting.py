import coremltools as ct
import tensorflow as tf

def convert_model(input_path='BreathingModel.h5', output_path='BreathingModel.mlmodel', input_shape=(1,3)):
    # Loading Model
    model = tf.keras.models.load_model(input_path, compile=False)

    # Converting into MLModel
    mlmodel = ct.convert(
              model,
              source="tensorflow",
              inputs=[ct.TensorType(input_shape)],
              minimum_deployment_target=ct.target.iOS14
    )

    # Saving the model in MLModel format
    mlmodel.save(output_path)

if __name__ == "__main__":
    convert_model()