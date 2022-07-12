# import dependencies
import numpy as np

def intensity(phi, tt, tm, tb, at, ab):
    A = tm * (tb * tt * at + ab)
    B = tm * (tt * at + tb * ab)
    C = tb + (tt * at * ab)
    D = 1 + (tb * tt * at * ab)
    E = (tt * at * ab) - tb
    F = (tb * tt * at * ab) - 1

    cfs_n = (np.cos(phi) * C - A) + ((np.sin(phi) * E) * 1j)
    cfs_d = (np.cos(phi) * D - B) + ((np.sin(phi) * F) * 1j)
    cfs = cfs_n / cfs_d
    ccfs = cfs.conjugate()

    return np.real(cfs * ccfs)

def intensityExplicit(phi, tt, tm, tb, at, ab):
    A = tm * (tb * tt * at + ab)
    B = tm * (tt * at + tb * ab)
    C = tb + (tt * at * ab)
    D = 1 + (tb * tt * at * ab)
    E = (tt * at * ab) - tb
    F = (tb * tt * at * ab) - 1

    i_n = C**2 * np.cos(phi)**2 + A**2 + E**2 * np.sin(phi)**2 - 2 * A * C * np.cos(phi)
    i_d = D**2 * np.cos(phi)**2 + B**2 + F**2 * np.sin(phi)**2 - 2 * B * D * np.cos(phi)

    return i_n / i_d