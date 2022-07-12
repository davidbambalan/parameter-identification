# import dependencies
import numpy as np
import cv2

def biggestContour(contours):
    # constants
    MIN_AREA = 5000
    EPSILON_COEFF = 0.02

    # declare variables
    biggest = np.array([])
    maxArea = 0

    # loop through every contour
    for contour in contours:
        contourArea = cv2.contourArea(contour)
        
        if (contourArea > MIN_AREA):
            contourPerimeter = cv2.arcLength(contour, True)
            contourShape = cv2.approxPolyDP(contour, EPSILON_COEFF * contourPerimeter, True)
            if (contourArea > maxArea) and (len(contourShape) == 4):
                biggest = contourShape
                maxArea = contourArea

    return biggest, maxArea


def drawRectangle(pts, img, color=(0, 255, 0), thickness = 1):
    # draw rectangle
    return cv2.polylines(img, [pts], True, color, thickness)

def reorder(pts):
    pts = pts.reshape((4,2))
    
    xs = np.sort(pts[:, 0])
    ys = np.sort(pts[:, 1])

    ptsTop = [[x,y] for [x,y] in pts if (y == ys[0] or y == ys[1])]
    ptsLeft = [[x,y] for [x,y] in pts if (x == xs[0] or x == xs[1])]

    ptTopLeft       = [[x,y] for [x,y] in pts if ([x,y] in ptsTop and [x,y] in ptsLeft)]
    ptTopRight      = [[x,y] for [x,y] in pts if ([x,y] in ptsTop and [x,y] not in ptsLeft)]
    ptBottomRight   = [[x,y] for [x,y] in pts if ([x,y] not in ptsTop and [x,y] not in ptsLeft)]
    ptBottomLeft    = [[x,y] for [x,y] in pts if ([x,y] not in ptsTop and [x,y] in ptsLeft)]

    return np.array([ptTopLeft, ptTopRight, ptBottomRight, ptBottomLeft]).reshape((4, 2))
