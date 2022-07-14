## import dependencies
import numpy as np
import cv2
import scipy.optimize as skopt
from PIL import ImageTk, Image

## import utilities
import utils.utils as utils
import utils.model as model

# constants
imgHeight = 1080
imgWidth = int(imgHeight * 4 / 3)

X_MIN = -np.pi
X_MAX = np.pi
Y_MIN = 0
Y_MAX = 1

def detectCorners(img):
    img = cv2.resize(img, (imgWidth, imgHeight))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)

    threshold1 = 18
    threshold2 = 18
    imgEdge = cv2.Canny(imgBlur, threshold1, threshold2)

    kernel = np.ones((4, 4)) # square kernel
    imgDilate = cv2.dilate(imgEdge, kernel, iterations=2)
    imgErode = cv2.erode(imgDilate, kernel, iterations=1)

    contours, _ = cv2.findContours(imgErode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # return cv2.drawContours(img, contours, -1, (0, 255, 0), 5)
    biggest, _ = utils.biggestContour(contours)
    return utils.drawRectangle(biggest, img, thickness=5), utils.reorder(np.float32(biggest)), imgErode

def convertToTk(___img):
    B, G, R = cv2.split(___img)
    __img = cv2.merge((R, G, B))
    __img = cv2.resize(__img, (396, 297))
    _img = Image.fromarray(__img)

    return ImageTk.PhotoImage(image=_img)

def warpImage(img, cornerCoords):
    img = cv2.resize(img, (imgWidth, imgHeight))
    targetCoords = np.float32([
        [0, 0],
        [imgWidth, 0],
        [imgWidth, imgHeight],
        [0, imgHeight]
    ])
    tform = cv2.getPerspectiveTransform(cornerCoords, targetCoords)
    imgWarped = cv2.warpPerspective(img, tform, (imgWidth, imgHeight))

    return imgWarped[0:imgHeight, 0:imgWidth]

def generatePoints(imgErode, cornerCoords):
    targetCoords = np.float32([
        [0, 0],
        [imgWidth, 0],
        [imgWidth, imgHeight],
        [0, imgHeight]
    ])
    tform = cv2.getPerspectiveTransform(cornerCoords, targetCoords)

    imgWarpedThreshold = cv2.warpPerspective(imgErode, tform, (imgWidth, imgHeight))
    # imgFinalThreshold = cv2.resize(imgWarpedThreshold[0:imgHeight, 0:imgWidth], (960, 720))
    curve =  utils.generateCurve(imgWarpedThreshold)

    return utils.interpolate(imgWarpedThreshold, curve, X_MIN, X_MAX, Y_MIN, Y_MAX)

def fitCurve(pts):
    popt, _ = skopt.curve_fit(model.intensityExplicit, pts[:, 0], pts[:, 1], bounds=(0, 1))
    return popt
