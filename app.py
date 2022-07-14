# import dependencies
import numpy as np
import cv2
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)

# import utilities
import utils.utils as utils
import utils.preprocess as preprocess
import utils.model as model

# constants
X_MIN = -np.pi
X_MAX = np.pi
Y_MIN = 0
Y_MAX = 1

def uploadFile():
    filename = utils.uploadFile()
    imgFilename.set(filename)
    
    global _img
    _img = cv2.imread(filename)
    img = preprocess.convertToTk(_img)
    lblImageOrig.configure(image=img)
    lblImageOrig.image = img

def detectCorners(img):
    global cornerCoords
    global imgErode
    imgCorners, cornerCoords, imgErode = preprocess.detectCorners(img)
    imgCorners = preprocess.convertToTk(imgCorners)
    lblImageOrig.configure(image=imgCorners)
    lblImageOrig.image = imgCorners

def convertImgToGraph(img, cornerCoords):
    global imgWarped
    imgWarped = preprocess.warpImage(img, cornerCoords)
    ax.clear()
    configurePlot()
    ax.imshow(imgWarped, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], aspect='auto')
    canvas.draw()

def configurePlot():
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_ylabel("Intensity")
    ax.set_xticks([X_MIN, 0.0, X_MAX])
    ax.set_xticklabels([r"$-\pi$", 0, r"$\pi$"])
    ax.grid(linestyle='--')

def detectPoints(imgErode, cornerCoords):
    global pts
    pts = preprocess.generatePoints(imgErode, cornerCoords)
    ax.clear()
    configurePlot()
    ax.imshow(imgWarped, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], aspect='auto')
    ax.scatter(pts[:, 0], pts[:, 1], label='data')
    ax.legend()
    canvas.draw()

def fitCurve(pts):
    popt = preprocess.fitCurve(pts)
    SAMPLES = 100
    phi = np.linspace(X_MIN, X_MAX, SAMPLES)
    ax.clear()
    configurePlot()
    ax.imshow(imgWarped, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], aspect='auto')
    ax.plot(phi, model.intensityExplicit(phi, *popt), color="red", label="model")
    ax.scatter(pts[:, 0], pts[:, 1], label='data')
    ax.legend()
    canvas.draw()

    lblTT['text'] = f' > tau_t: \t\t {popt[0]}'
    lblTM['text'] = f' > tau_m: \t {popt[1]}'
    lblTB['text'] = f' > tau_b: \t {popt[2]}'
    lblAT['text'] = f' > alpha_t: \t {popt[3]}'
    lblAB['text'] = f' > alpha_b: \t {popt[4]}'

# WINDOW
# init window
root = tk.Tk()
root.resizable(False, False)
root.geometry('960x720')
root.title('Parameter Identification')
# TODO: icon
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(1, weight=1)

# IMPORT FRAME
frImport = tk.LabelFrame(
    master = root,
    text = " Step 1: Upload image ",
    padx = 24,
    pady = 24
)
frImport.grid(column=0, row=0, columnspan=2, padx=24, pady=24, sticky=tk.EW)
frImport.grid_columnconfigure(1, weight=1)
    
btnImport = tk.Button(
    master = frImport,
    text = 'Upload',
    padx = 24,
    pady = 0,
    command = uploadFile
)
btnImport.grid(column=0, row=0, padx=(0, 12))

imgFilename = tk.StringVar()
txtImport = tk.Entry(
    master = frImport,
    state = 'disabled',
    textvariable = imgFilename
)
txtImport.grid(column=1, row=0, sticky=tk.EW)

# LEFT FRAME
frLeft = tk.LabelFrame(
    master = root,
    text = " Step 2: Detect Graph ",
    padx = 24,
    pady = 24
)
frLeft.grid(column=0, row=1, padx=(24, 12), pady=(0, 24), sticky=tk.NSEW)
frLeft.grid_columnconfigure(0, weight=1)

frImageOrig = tk.Frame(
    master = frLeft,
    bg = '#CCCCCC',
    width = 396, 
    height = 297
)
frImageOrig.grid(column=0, row=0, columnspan=2, sticky=tk.W)
frImageOrig.grid_propagate(False)

lblImageOrig = tk.Label(
    master = frImageOrig,
    bg = '#CCCCCC'
)
lblImageOrig.place(x=-2, y=-2)

btnDetect = tk.Button(
    master = frLeft,
    text = 'Auto-detect corners',
    padx = 24,
    pady = 0,
    command = lambda: detectCorners(_img)
)
btnDetect.grid(column=0, row=1, pady=(12, 0), sticky=tk.E)

btnToGraph = tk.Button(
    master = frLeft,
    text = 'Convert to graph',
    padx = 24,
    pady = 0,
    command = lambda: convertImgToGraph(_img, cornerCoords)
)
btnToGraph.grid(column=1, row=1, pady=(12, 0), padx=(12, 0), sticky=tk.E)

# RIGHT FRAME
frRight = tk.LabelFrame(
    master = root,
    text = " Step 3: Fit Curve ",
    padx = 24,
    pady = 24
)
frRight.grid(column=1, row=1, padx=(12, 24), pady=(0, 24), sticky=tk.NSEW)
frRight.grid_columnconfigure(0, weight=1)

frGraph = tk.Frame(
    master = frRight,
    bg = '#CCCCCC',
    width = 396,
    height = 297
)
frGraph.grid(column=0, row=0, columnspan=2, sticky=tk.W)

fig = plt.Figure(figsize=(4, 3), dpi=100, constrained_layout=True)
ax = fig.add_subplot(111)
configurePlot()
canvas = FigureCanvasTkAgg(fig, master = frGraph)
canvas.draw()
canvas.get_tk_widget().pack()

btnDetectPoints = tk.Button(
    master = frRight,
    text = 'Detect points',
    padx = 24,
    pady = 0,
    command = lambda: detectPoints(imgErode, cornerCoords)
)
btnDetectPoints.grid(column=0, row=1, pady=(12, 0), sticky=tk.E)

btnCurveFit = tk.Button(
    master = frRight,
    text = 'Fit curve',
    padx = 24,
    pady = 0,
    command = lambda: fitCurve(pts)
)
btnCurveFit.grid(column=1, row=1, pady=(12, 0), padx=(12, 0), sticky=tk.E)

lblParams = tk.Label(
    master = frRight,
    text = 'Model parameters:'
)
lblParams.grid(column=0, row=2, sticky=tk.W)

lblTT = tk.Label(
    master = frRight,
    text = f' > tau_t: \t\t 0'
)
lblTT.grid(column=0, row=3, sticky=tk.W)

lblTM = tk.Label(
    master = frRight,
    text = f' > tau_m: \t 0'
)
lblTM.grid(column=0, row=4, sticky=tk.W)

lblTB = tk.Label(
    master = frRight,
    text = f' > tau_b: \t 0'
)
lblTB.grid(column=0, row=5, sticky=tk.W)

lblAT = tk.Label(
    master = frRight,
    text = f' > alpha_t: \t 0'
)
lblAT.grid(column=0, row=6, sticky=tk.W)

lblAB = tk.Label(
    master = frRight,
    text = f' > alpha_b: \t 0'
)
lblAB.grid(column=0, row=7, sticky=tk.W)


# run window
root.mainloop()