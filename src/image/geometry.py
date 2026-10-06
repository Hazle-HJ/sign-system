import cv2
import numpy as np


def analyze_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges > 0)

    return {"edge_density": round(float(edge_density), 3)}
