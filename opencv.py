import cv2
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt


def check_expect(real, test):
    if real.all() == test.all():
        print("Test passed")
    else:
        print(f"Error!!! {real} != {test}")


def open_img(path=""):
    path_to_file = path or input("Enter path to image: \t")
    img = cv2.imread(path_to_file, cv2.IMREAD_COLOR)
    if img is None:
        print("Can't open image:(")
        return None
    cv2.imshow("Your image", img)
    # cv2.waitKey(0)
    return img


def plot(dataset, title="", xlabel="", ylabel=""):
    plt.plot(range(dataset.size), dataset)
    plt.title = title
    plt.xlabel = xlabel
    plt.ylabel = ylabel
    plt.grid()
    plt.show()


def color_to_bw(img, color_component=0):
    """
    color_component:
        0 - red
        1 - green
        2 - blue
    """

    img_gray = img.copy()
    match (color_component):
        case 0:
            for i in range(img_gray.shape[0]):
                for j in range(img_gray.shape[1]):
                    (_, _, r) = img_gray[i, j]
                    gray = 0 if r <= 256 / 2 else 255
                    img_gray[i, j] = gray
        case 1:
            for i in range(img_gray.shape[0]):
                for j in range(img_gray.shape[1]):
                    (_, g, _) = img_gray[i, j]
                    gray = 0 if g <= 256 / 2 else 255
                    img_gray[i, j] = gray
        case 2:
            for i in range(img_gray.shape[0]):
                for j in range(img_gray.shape[1]):
                    (b, _, _) = img_gray[i, j]
                    gray = 0 if b <= 256 / 2 else 255
                    img_gray[i, j] = gray

    return img_gray


def color_to_gray(img):
    img_gray = img.copy()

    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            (r, g, b) = img_gray[i, j, ::-1]
            gray = int((max(r, g, b) + min(r, g, b)) / 2)
            img_gray[i, j] = gray % 255

    return img_gray


def vertical_proection(img):
    if img.size == img.shape[0] * img.shape[1]:
        v_pr = np.zeros(img.shape[1])
        for i in range(img.shape[1]):
            for j in range(img.shape[0]):
                v_pr[i] += img[j, i]

        v_pr = v_pr.astype(int)
    else:
        arr1 = vertical_proection(img[:, :, 0])
        arr2 = vertical_proection(img[:, :, 1])
        arr3 = vertical_proection(img[:, :, 2])
        print(f"arr1 sizes: {arr1.size}")
        print(f"arr1 sizes: {arr2.size}")
        print(f"arr1 sizes: {arr3.size}")
        v_pr = np.zeros((arr1.size, 3))
        for i in range(arr1.size):
            v_pr[i, 0] = arr1[i]
            v_pr[i, 1] = arr2[i]
            v_pr[i, 2] = arr3[i]

    return v_pr


# produce matrix with rows HEIGHT (img.shape[1]) and cols (img.shape[2])
# if its 3 colored, then colors will be in (b, g, r)
def horizontal_proection(img):
    if img.size == img.shape[0] * img.shape[1]:
        h_pr = np.zeros(img.shape[0])
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                h_pr[i] += img[i, j]

        h_pr = h_pr.astype(int)
    else:
        arr1 = horizontal_proection(img[:, :, 0])
        arr2 = horizontal_proection(img[:, :, 1])
        arr3 = horizontal_proection(img[:, :, 2])
        print(f"arr1 sizes: {arr1.size}")
        print(f"arr1 sizes: {arr2.size}")
        print(f"arr1 sizes: {arr3.size}")
        h_pr = np.zeros((arr1.size, 3))
        for i in range(arr1.size):
            h_pr[i, 0] = arr1[i]
            h_pr[i, 1] = arr2[i]
            h_pr[i, 2] = arr3[i]

    return h_pr


class MODE(Enum):
    TEST = 0
    RUN = 1


def client(mode):
    if mode == MODE.TEST:
        img = open_img("cofee.png")
    else:
        img = open_img()

    if img is None:
        return

    bw_image = color_to_bw(img, 2)
    gray_img = color_to_gray(img)

    cv2.imshow("BW_image (blue_component)", bw_image)
    cv2.imshow("gray_image", gray_img)
    # cv2.waitKey(0)

    # Расчитать ряды данных вертикальных и горизонатльных проекций

    hp_res_for_gray = horizontal_proection(gray_img)
    plot(hp_res_for_gray[:, 0], "HFrequency for gray")

    vp_res_for_gray = vertical_proection(gray_img)
    plot(vp_res_for_gray[:, 0], title="VFrequency for gray")



if __name__ == "__main__":
    client(MODE.TEST)
