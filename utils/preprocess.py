## import dependencies
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.optimize as skopt
import tkinter as tk
from PIL import ImageTk, Image

## import utilities
import utils.utils as utils
import utils.model as model

# # IMAGE PREPROCESSING
# ## Read Image
# ### read image
imgPath = './assets/image.jpg'
_img = cv2.imread(imgPath)
img = _img

# # ### resize image
imgHeight = 1080
imgWidth = int(imgHeight * 4 / 3)

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
    return utils.drawRectangle(biggest, img, thickness=5)

def convertToTk(___img):
    B, G, R = cv2.split(___img)
    __img = cv2.merge((R, G, B))
    __img = cv2.resize(__img, (396, 297))
    _img = Image.fromarray(__img)

    return ImageTk.PhotoImage(image=_img)

def warpImage(img):
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
    biggest, _ = utils.biggestContour(contours)

    cornerCoords = utils.reorder(np.float32(biggest))
    targetCoords = np.float32([
        [0, 0],
        [imgWidth, 0],
        [imgWidth, imgHeight],
        [0, imgHeight]
    ])

    tform = cv2.getPerspectiveTransform(cornerCoords, targetCoords)
    imgWarped = cv2.warpPerspective(img, tform, (imgWidth, imgHeight))

    return imgWarped[0:imgHeight, 0:imgWidth]


# imgWarpedThreshold = cv2.warpPerspective(imgErode, tform, (imgWidth, imgHeight))
# imgFinalThreshold = cv2.resize(imgWarpedThreshold[0:imgHeight, 0:imgWidth], (finalWidth, finalHeight))

# # Generate Points
# curve = utils.generateCurve(imgFinalThreshold)
# # interpolate
# X_MIN = -np.pi
# X_MAX = np.pi
# Y_MIN = 0
# Y_MAX = 1
# pts = utils.interpolate(imgFinalThreshold, curve, X_MIN, X_MAX, Y_MIN, Y_MAX)

# # Get Parameters
# popt, _ = skopt.curve_fit(model.intensityExplicit, pts[:, 0], pts[:, 1], bounds=(0, 1))

# # Generate Figure
# fig, ax = plt.subplots()
# ax.set_xlim(X_MIN, X_MAX)
# ax.set_ylim(Y_MIN, Y_MAX)
# ax.set_ylim(Y_MIN, Y_MAX)
# ax.set_ylabel("Intensity")
# ax.set_xticks([X_MIN, 0.0, X_MAX])
# ax.set_xticklabels([r"$-\pi$", 0, r"$\pi$"])
# ax.grid(linestyle='--')
# SAMPLES = 1000
# phi = np.linspace(X_MIN, X_MAX, SAMPLES)
# ax.imshow(imgFinal, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], aspect='auto')
# ax.scatter(pts[:, 0], pts[:, 1], label='data')
# ax.plot(phi, model.intensityExplicit(phi, *popt), color="red", label="model")
# ax.legend()
# print(f"tt: {popt[0]}, tm: {popt[1]}, tb: {popt[2]}, at: {popt[3]}, ab: {popt[4]}")
# # plt.show()