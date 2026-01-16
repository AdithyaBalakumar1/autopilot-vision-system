"""
Lane Detection Module

Implements a classical computer vision pipeline for detecting lane lines
in road images using OpenCV.
"""

import cv2
import numpy as np

def region_of_interest(edges):
    """
    Apply a mask to keep only the region of the image
    where lane lines are expected.
    """
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # Define a triangular region of interest
    polygon = np.array([[
        (0, height),
        (width // 2, height // 2),
        (width, height)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges

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
    edges = region_of_interest(edges)


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
