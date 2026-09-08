import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

TFLITE_MODEL = "tf_model/mini_impact_cnn_float32.tflite"
PATCH_PATH = "data/patches/images/0.png"


def test_tflite():
    interpreter = tflite.Interpreter(model_path=TFLITE_MODEL)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = cv2.imread(PATCH_PATH, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    img = img[np.newaxis, np.newaxis, :, :]  # (1, 1, 64, 64)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    logits = interpreter.get_tensor(output_details[0]["index"])[0]
    prob = np.exp(logits) / np.sum(np.exp(logits))

    print("Logits :", logits)
    print("Prob impact :", float(prob[1]))


if __name__ == "__main__":
    test_tflite()
