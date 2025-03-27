import coremltools as ct
import tensorflow as tf
import os
import sys

def convert_model(input_path='BreathingModel.h5', output_path='BreathingModel.mlmodel', input_shape=(1,3)):
    # Loading Model
    print("Loading Model...")
    if not os.path.exists(input_path):
        print(f"File does not exist：{input_path}")
        sys.exit(1)


    try:
        model = tf.keras.models.load_model(input_path, compile=False)
    except Exception as e:
        print(f"Model loading failed: {e}")
        sys.exit(1)


    # Converting into MLModel
    print("Start converting...")
    try:
        mlmodel = ct.convert(
                  model,
                  source="tensorflow",
                  inputs=[ct.TensorType(shape=input_shape)],
                  minimum_deployment_target=ct.target.iOS14
        )

        # Saving the model in MLModel format

        try:
            print("Converting successful.")
            mlmodel.save(output_path)
            print("The mlmodel format model is saved.")

        except Exception as e:
            print(f"Saving failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Converting failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    convert_model()