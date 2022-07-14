# import dependencies
import numpy as np
import cv2

import numpy as np
import cv2
import tkinter.filedialog

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

def generateCurve(img):
    # constants
    MIN_LENGTH = 1000
    EPSILON_COEFF = 0.001
    IMG_HEIGHT, IMG_WIDTH = img.shape

    # declare variables
    biggest = np.array([])
    maxLength = 0

    curveContours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in curveContours:
        contourLength = cv2.arcLength(contour, False)
        if (contourLength > MIN_LENGTH) and (contourLength > maxLength):
            contourShape = cv2.approxPolyDP(contour, EPSILON_COEFF * contourLength, False)
            biggest = contourShape
            maxLength = contourLength

    # eliminate outliers
    length, _, _ = biggest.shape
    biggest = biggest.reshape((length, 2))
    biggest = np.array([[x, y] for [x, y] in biggest if (x > 0.05*IMG_WIDTH and x < 0.95*IMG_WIDTH) and (y > 0.05*IMG_HEIGHT and y < 0.95*IMG_HEIGHT)])

    return biggest.reshape((len(biggest), 1, 2))

def interpolate(img, pts, xMin=-np.pi, xMax=np.pi, yMin=0, yMax=1):
    # constant
    IMG_HEIGHT, IMG_WIDTH = img.shape

    # declare variales
    newPts = []

    length, _, _ = pts.shape
    pts = pts.reshape((length, 2))

    for pt in pts:
        _x = pt[0]
        _y = pt[1]
        x = xMin + ((_x/IMG_WIDTH) * (xMax - xMin))
        y = yMax - ((_y/IMG_HEIGHT) * (yMax - yMin))
        newPts.append([x, y])

    return np.array(newPts)

def uploadFile():
    filename = tkinter.filedialog.askopenfilename(
        initialdir = './',
        title = 'Select image',
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg")
        ]
    )

    return filename