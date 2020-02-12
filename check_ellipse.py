import cv2
from utility_functions import *
import warnings, traceback
warnings.filterwarnings("error")

#### Functions to evaluate the shape of grain as an Ellipse ####

def get_chain_code(boundry):
    chainCode = [0]
    p1 = boundry[0]
    for p2 in boundry[1:]:
        if p2[0] - p1[0] == 0  and p2[1] - p1[1] == 1:
            chainCode.append(0)
        elif p2[0] - p1[0] == 1  and p2[1] - p1[1] == 1:
            chainCode.append(7)
        elif p2[0] - p1[0] == 1  and p2[1] - p1[1] == 0:
            chainCode.append(6)
        elif p2[0] - p1[0] == 1  and p2[1] - p1[1] == -1:
            chainCode.append(5)
        elif p2[0] - p1[0] == 0  and p2[1] - p1[1] == -1:
            chainCode.append(4)
        elif p2[0] - p1[0] == -1  and p2[1] - p1[1] == -1:
            chainCode.append(3)
        elif p2[0] - p1[0] == -1  and p2[1] - p1[1] == 0:
            chainCode.append(2)
        elif p2[0] - p1[0] == -1  and p2[1] - p1[1] == 1:
            chainCode.append(1)
        p1=p2
    return chainCode

def elliptic_fourier_descriptors(contour, order=10):
    dxy = np.diff(contour, axis=0)
    dt = np.sqrt((dxy ** 2).sum(axis=1))
    t = np.concatenate([([0., ]), np.cumsum(dt)])
    T = t[-1]
    try:
        phi = (2 * np.pi * t) / T
    except RuntimeWarning:
        traceback.print_exc()
        return None

    coeffs = np.zeros((order, 4))
    for n in range(1, order + 1):
        const = T / (2 * n * n * np.pi * np.pi)
        phi_n = phi * n
        d_cos_phi_n = np.cos(phi_n[1:]) - np.cos(phi_n[:-1])
        d_sin_phi_n = np.sin(phi_n[1:]) - np.sin(phi_n[:-1])
        a_n = const * np.sum((dxy[:, 0] / dt) * d_cos_phi_n)
        b_n = const * np.sum((dxy[:, 0] / dt) * d_sin_phi_n)
        c_n = const * np.sum((dxy[:, 1] / dt) * d_cos_phi_n)
        d_n = const * np.sum((dxy[:, 1] / dt) * d_sin_phi_n)
        coeffs[n - 1, :] = a_n, b_n, c_n, d_n
    return coeffs

def efd(coeffs,contour_1,  locus=(0., 0.)):

    N = coeffs.shape[0]
    N_half = int(np.ceil(N / 2))
    n_rows = 2
    n = len(contour_1)

    t = np.linspace(0, 1.0, n)
    xt = np.ones((n,)) * locus[0]
    yt = np.ones((n,)) * locus[1]

    for n in range(coeffs.shape[0]):
        xt += (coeffs[n, 0] * np.cos(2 * (n + 1) * np.pi * t)) + \
              (coeffs[n, 1] * np.sin(2 * (n + 1) * np.pi * t))
        yt += (coeffs[n, 2] * np.cos(2 * (n + 1) * np.pi * t)) + \
              (coeffs[n, 3] * np.sin(2 * (n + 1) * np.pi * t))
    return xt,yt
