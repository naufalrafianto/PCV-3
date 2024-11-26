import cv2
import numpy as np


def detect_card(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_preview = np.zeros_like(edged)
    cv2.drawContours(contour_preview, contours, -1, (255, 255, 255), 2)

    if len(contours) > 0:
        card_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(card_contour) > 5000:
            peri = cv2.arcLength(card_contour, True)
            approx = cv2.approxPolyDP(card_contour, 0.02 * peri, True)

            if len(approx) == 4:
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                corners = approx.reshape(4, 2)

                for corner in corners:
                    x, y = corner.astype(int)
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

                return True, corners, contour_preview, edged

    return False, None, contour_preview, edged
