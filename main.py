import argparse
import numpy as np
from skimage.io import imread, imsave
from numpy.typing import NDArray
from scipy.signal import gaussian
from scipy.ndimage import convolve
from typing import List


def read_image(filename: str) -> NDArray:
    return imread(filename).astype(np.float32)


def save_image(filename: str, image: NDArray) -> None:
    imsave(filename, np.clip(np.round(image), 0, 255).astype(np.uint8))


def get_gauss2d_dx(rad: int, sigma: float) -> NDArray:
    gauss1d_dx = gaussian(2 * rad + 1, sigma) * (-np.arange(-rad, rad + 1) /
                                                 sigma ** 2)
    return np.outer(gauss1d_dx, gaussian(2 * rad + 1, sigma))


def get_gauss2d_dy(rad: int, sigma: float) -> NDArray:
    gauss1d_dx = gaussian(2 * rad + 1, sigma) * (-np.arange(-rad, rad + 1) /
                                                 sigma ** 2)
    return np.outer(gaussian(2 * rad + 1, sigma), gauss1d_dx)


def gradient(image: NDArray, sigma: float) -> List[NDArray]:
    is_colored = False
    if image.ndim == 3:
        is_colored = True
        image = image[:, :, 0]

    rad = int(3 * sigma)

    gauss2d_dx = get_gauss2d_dx(rad, sigma)
    gauss2d_dy = get_gauss2d_dy(rad, sigma)

    grad_dx = convolve(image, gauss2d_dx, mode="nearest")
    grad_dy = convolve(image, gauss2d_dy, mode="nearest")

    grad_mod = np.sqrt(grad_dx ** 2 + grad_dy ** 2)
    grad_mod = grad_mod / np.max(grad_mod) * 255.0

    if is_colored:
        grad_mod = np.stack([grad_mod, grad_mod, grad_mod], axis=2)

    grad_dir = np.arctan2(grad_dy, grad_dx) * 180 / np.pi
    grad_dir[grad_dir < 0] += 180
    return [grad_mod, grad_dir]


def nonmax(image: NDArray, sigma: float) -> NDArray:
    is_colored = False
    if image.ndim == 3:
        is_colored = True
        image = image[:, :, 0]

    grad_mod, grad_dir = gradient(image, sigma)
    grad_dir = 45 * (np.round(grad_dir / 45))
    res = np.zeros_like(grad_mod)

    for i in range(grad_mod.shape[0]):
        for j in range(grad_mod.shape[1]):
            angle = grad_dir[i, j]
            p = r = 255
            if angle == 180 or angle == 0:
                if i == 0:
                    p = r = grad_mod[i + 1, j]
                elif i == grad_mod.shape[0] - 1:
                    p = r = grad_mod[i - 1, j]
                else:
                    p, r = grad_mod[i + 1, j], grad_mod[i - 1, j]
            elif angle == 90:
                if j == 0:
                    p = r = grad_mod[i, j + 1]
                elif j == grad_mod.shape[1] - 1:
                    p = r = grad_mod[i, j - 1]
                else:
                    p, r = grad_mod[i, j + 1], grad_mod[i, j - 1]
            elif angle == 45:
                if i == 0:
                    if j != grad_mod.shape[1] - 1:
                        p = r = grad_mod[i + 1, j + 1]
                elif i == grad_mod.shape[0] - 1:
                    if j != 0:
                        p = r = grad_mod[i - 1, j - 1]
                else:
                    if j == 0:
                        if i != grad_mod.shape[0] - 1:
                            p = r = grad_mod[i + 1, j + 1]
                    elif j == grad_mod.shape[1] - 1:
                        if i != 0:
                            p = r = grad_mod[i - 1, j - 1]
                    else:
                        p, r = grad_mod[i + 1, j + 1], grad_mod[i - 1, j - 1]
            else:
                if i == 0:
                    if j != 0:
                        p = r = grad_mod[i + 1, j - 1]
                elif i == grad_mod.shape[0] - 1:
                    if j != grad_mod.shape[1] - 1:
                        p = r = grad_mod[i - 1, j + 1]
                else:
                    if j == 0:
                        if i != 0:
                            p = r = grad_mod[i - 1, j + 1]
                    elif j == grad_mod.shape[1] - 1:
                        if i != grad_mod.shape[0] - 1:
                            p = r = grad_mod[i - 1, j - 1]
                    else:
                        p, r = grad_mod[i + 1, j - 1], grad_mod[i - 1, j + 1]

            if p < grad_mod[i, j] and r < grad_mod[i, j]:
                res[i, j] = grad_mod[i, j]

    if is_colored:
        res = np.stack([res, res, res], axis=2)

    return res


def hysteresis(image: NDArray, weak_val: int):
    res = image.copy()
    strong_x, strong_y = np.where(image == 255)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0),
                  (1, 1)]
    while len(strong_x):
        x, y = strong_x[0], strong_y[0]
        strong_x = np.delete(strong_x, 0)
        strong_y = np.delete(strong_y, 0)

        for direction in directions:
            new_x, new_y = x + direction[0], y + direction[1]
            if (0 <= new_x < image.shape[0] and 0 <= new_y < image.shape[1] and
               res[new_x, new_y] == weak_val):
                res[new_x, new_y] = 255
                strong_x = np.append(strong_x, new_x)
                strong_y = np.append(strong_y, new_y)
    res[res != 255] = 0
    return res


def canny(image: NDArray, sigma: float, thr_high: float,
    thr_low: float) -> NDArray:

    is_colored = False
    if image.ndim == 3:
        is_colored = True
        image = image[:, :, 0]

    edges = nonmax(image, sigma)

    high_threshold = np.max(edges) * thr_high
    low_threshold = np.max(edges) * thr_low

    map = np.zeros_like(edges, dtype=int)
    weak_val = 100

    map[edges >= high_threshold] = 255
    map[(low_threshold < edges) & (edges < high_threshold)] = weak_val

    res = hysteresis(map, weak_val)

    if is_colored:
        res = np.stack([res, res, res], axis=2)

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    parser_grad = subparsers.add_parser("grad", help="Calculates modulus of"
                                                     "gradient")
    parser_grad.add_argument("sigma", type=float)

    parser_nonmax = subparsers.add_parser("nonmax", help="Non-maximal"
                                                         "suppression")
    parser_nonmax.add_argument("sigma", type=float)

    parser_canny = subparsers.add_parser("canny", help="Canny edge detecting")
    parser_canny.add_argument("sigma", type=float)
    parser_canny.add_argument("thr_high", type=float)
    parser_canny.add_argument("thr_low", type=float)

    parser_mse = subparsers.add_parser("mse")

    parser.add_argument("input_image", type=str)
    parser.add_argument("output_image", type=str)

    args = parser.parse_args()

    if args.command == "grad":
        image = read_image(args.input_image)
        grad_mod = gradient(image, args.sigma)[0]
        save_image(args.output_image, grad_mod)
    elif args.command == "nonmax":
        image = read_image(args.input_image)
        edges = nonmax(image, args.sigma)
        save_image(args.output_image, edges)
    elif args.command == "canny":
        image = read_image(args.input_image)
        res = canny(image, args.sigma, args.thr_high, args.thr_low)
        save_image(args.output_image, res)
