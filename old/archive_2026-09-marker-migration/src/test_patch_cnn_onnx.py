import onnxruntime as ort
import numpy as np
import cv2

PATCH_PATH = "data/patches/images/0.png"
ONNX_PATH = "mini_impact_cnn.onnx"
TFLITE_MODEL = "tf_model/mini_impact_cnn_float32.tflite"


def test_onnx():
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    img = cv2.imread(PATCH_PATH, cv2.IMREAD_GRAYSCALE)
    img = img.astype("float32") / 255.0
    img = img[None, None, :, :]  # (1, 1, 64, 64)

    outputs = sess.run(None, {"input": img})
    logits = outputs[0][0]

    prob = np.exp(logits) / np.sum(np.exp(logits))
    print("Logits :", logits)
    print("Prob impact :", float(prob[1]))


if __name__ == "__main__":
    test_onnx()
