"""
Lane Detection Module

Implements a classical computer vision pipeline for detecting lane lines
in road images using OpenCV.
"""

import cv2
import numpy as np


def detect_lanes(frame):
    """
    Detect lane lines from an input frame.

    Args:
        frame (np.ndarray): BGR image from OpenCV

    Returns:
        lines (np.ndarray): Detected lane line segments
    """

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Detect line segments using Hough Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=100,
        maxLineGap=50
    )

    return lines
