import cv2
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from math import sqrt


def open_img(path=""):
    path_to_file = path or input("Enter path to image: \t")
    img = cv2.imread(path_to_file, cv2.IMREAD_COLOR)
    if img is None:
        print("Can't open image:(")
        return None
    # cv2.imshow("Your image", img)
    # cv2.waitKey(0)
    return img


def calc_energy(img: np.ndarray) -> np.ndarray:
    """
    :param img: изначальное изображение (трехцветное)
    :return: out_arr - карта энергий кажого пикселя
    """
    hl, vl, _ = img.shape

    out_arr: np.ndarray = np.zeros((hl, vl))

    for i in range(hl):
        for j in range(vl):
            if i == 0:
                (mb, mg, mr) = img[i, j].astype(np.int32)
                (rb, rg, rr) = img[i + 1, j].astype(np.int32)
                out_arr[i, j] = sqrt(
                    sum([pow(mr, 2), pow(mg, 2), pow(mb, 2), pow(rr - mr, 2), pow(rg - mg, 2), pow(rb - mb, 2)]))
            elif i == hl - 1:
                (mb, mg, mr) = img[i, j].astype(np.int32)
                (lb, lg, lr) = img[i - 1, j].astype(np.int32)
                out_arr[i, j] = sqrt(
                    sum([pow(lr - mr, 2), pow(lg - mg, 2), pow(lb - mb, 2), pow(mr, 2), pow(mg, 2), pow(mb, 2)]))
            else:
                (mb, mg, mr) = img[i, j].astype(np.int32)
                (rb, rg, rr) = img[i + 1, j].astype(np.int32)
                (lb, lg, lr) = img[i - 1, j].astype(np.int32)
                out_arr[i, j] = sqrt(
                    sum([pow(lr - mr, 2), pow(lg - mg, 2), pow(lb - mb, 2), pow(rr - mr, 2), pow(rg - mg, 2),
                         pow(rb - mb, 2)]))

    return out_arr


def camulative_map(energy: np.ndarray) -> tuple:
    rows, cols = energy.shape
    M = np.copy(energy)
    backtrack = np.zeros_like(M, dtype=np.int32)
    for i in range(1, rows):
        for j in range(cols):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i - 1, idx + j]
            elif j == cols - 1:
                idx = np.argmin(M[i - 1, j - 1:j + 1])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack


def find_seam(M, backtrack) -> np.ndarray:
    rows, cols = M.shape
    seam = np.zeros(rows, dtype=np.int32)
    seam[-1] = np.argmin(M[-1])
    for i in range(rows - 2, -1, -1):
        seam[i] = backtrack[i + 1, seam[i + 1]]
    return seam


def remove_seam(img, seam) -> np.ndarray:
    rows, cols, channels = img.shape
    new_img = np.zeros((rows, cols - 1, channels), dtype=img.dtype)
    for i in range(rows):
        if i >= len(seam):
            break
        j = seam[i]
        new_img[i, :, :] = np.delete(img[i, :, :], j, axis=0)
    return new_img


def paint_seam(img: np.ndarray, seam: np.ndarray, is_horizontal: bool = True) -> np.ndarray:
    painted_img = img.copy()
    rows = seam.shape[0]
    if is_horizontal:
        for i in range(rows):
            painted_img[seam[i], i] = (0, 0, 255)
    else:
        for i in range(rows):
            painted_img[i, seam[i]] = (0, 0, 255)
    return painted_img


def client():
    # Открываем изображение
    img = open_img("cofee.png")
    h, w, _ = img.shape
    print(f"Orginal image resolution {w}x{h}")

    # Вводим желаемый размер изображения
    width = int(input("Введите желаемую ширину картинки: "))
    height = int(input("Введите желаемую высоту картинки: "))

    # Уменьшаем высоту
    modified_img = img.copy()
    for i in range(w - width):
        energy = calc_energy(modified_img)
        M, backtrack = camulative_map(energy)
        seam = find_seam(M, backtrack)
        if i < 3:
            img_with_seam = paint_seam(modified_img, seam, False)
            cv2.imshow(f"{i + 1} height", img_with_seam)
        modified_img = remove_seam(modified_img, seam)

    # Уменьшаем ширину
    modified_img = modified_img.transpose((1, 0, 2))
    for i in range(h - height):
        energy = calc_energy(modified_img)
        M, backtrack = camulative_map(energy)
        seam = find_seam(M, backtrack)
        if i < 3:
            img_with_seam = paint_seam(modified_img.transpose((1,0,2)), seam)
            cv2.imshow(f"{i + 1} width", img_with_seam)
        modified_img = remove_seam(modified_img, seam)
    modified_img = modified_img.transpose((1, 0, 2))

    print(f"Produced image resolution"
          f" {modified_img.shape[1]}x{modified_img.shape[0]}")

    cv2.imwrite("Produced_image.jpg", modified_img)
    cv2.imshow("Original image", img)
    cv2.imshow("Produced image", modified_img)
    cv2.waitKey(0)


class Tests:
    pass


if __name__ == "__main__":
    client()
