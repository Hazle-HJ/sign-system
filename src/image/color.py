import cv2
import numpy as np


def analyze_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    saturation = hsv[:, :, 1]
    brightness = hsv[:, :, 2]

    return {
        "mean_saturation": round(float(np.mean(saturation)) / 255, 3),
        "mean_brightness": round(float(np.mean(brightness)) / 255, 3),
        "contrast": round(float(np.std(brightness)) / 255, 3),
    }
