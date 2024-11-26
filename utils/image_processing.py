import cv2
import numpy as np


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def auto_rotate_points(points):
    ordered = order_points(points)

    height_left = np.linalg.norm(ordered[3] - ordered[0])
    height_right = np.linalg.norm(ordered[2] - ordered[1])
    height = max(height_left, height_right)

    width_top = np.linalg.norm(ordered[1] - ordered[0])
    width_bottom = np.linalg.norm(ordered[2] - ordered[3])
    width = max(width_top, width_bottom)

    if width > height:
        new_ordered = np.array([
            ordered[3],
            ordered[0],
            ordered[1],
            ordered[2]
        ])
        return new_ordered

    return ordered


def get_warped_card(frame, corners, width=500, height=700):
    ordered_corners = auto_rotate_points(corners)

    current_width = np.linalg.norm(ordered_corners[1] - ordered_corners[0])
    current_height = np.linalg.norm(ordered_corners[3] - ordered_corners[0])

    if current_width > current_height:
        width, height = height, width

    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(ordered_corners, dst_points)
    warped = cv2.warpPerspective(frame, matrix, (width, height))

    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    binary_warped = cv2.adaptiveThreshold(
        gray_warped,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # Memastikan background hitam dan kartu putih
    if np.mean(binary_warped[0:50, 0:50]) > 127:
        binary_warped = cv2.bitwise_not(binary_warped)

    # Invert untuk mendapatkan kartu putih
    binary_warped = cv2.bitwise_not(binary_warped)

    return warped, binary_warped
