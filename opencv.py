from __future__ import annotations
from abc import ABC, abstractmethod

import cv2
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


class MyFig():

    def __init__(self, r: int, c: int):
        """
        :param r: count of rows on figure
        :param c: count of colums on figure
        """

        self.fig, self.ax = plt.subplots(r, c, gridspec_kw={'height_ratios': [1 for i in range(r)],
                                                            'width_ratios': [1 for i in range(c)]})

    def get_ax(self, i: int, j: int):
        return self.ax[i, j]

    def show(self):
        plt.show()


def open_img(path: str = "") -> np.ndarray:
    path_to_file = path or input("Enter path to image: \t")
    img = cv2.imread(path_to_file, cv2.IMREAD_COLOR)
    if img is None:
        print("Can't open image:(")
        return None

    return img


def plot(dataset: np.ndarray, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    plt.plot(range(dataset.size), dataset)
    plt.title = title
    plt.xlabel = xlabel
    plt.ylabel = ylabel
    plt.grid()
    plt.show()


def unite_plot_vp_and_hp(img: np.ndarray, hp: np.ndarray, vp: np.ndarray) -> None:
    plt.subplot(2, 2, 1)
    plt.imshow(img)

    plt.subplot(2, 2, 2)
    plt.plot(hp, range(hp.size))
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    # plt.plot(range(hp.size), hp)
    plt.legend("Horizontal projection")
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(range(vp.size), vp)
    plt.legend("Vertical projection")
    plt.xlim((0, vp.size))
    plt.grid()

    plt.show()


def unite_plot_new(fig: MyFig, img: np.darray, hp: np.ndarray, vp: np.ndarray) -> MyFig:
    ax = fig.get_ax(0, 0)
    ax.imshow(img)

    ax = fig.get_ax(0, 1)
    ax.plot(hp, range(hp.size))
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.legend("H")
    ax.grid()

    ax = fig.get_ax(1, 0)
    ax.plot(range(vp.size), vp)
    ax.legend("Vertical projection")
    ax.set_xlim((0, vp.size))
    ax.grid()

    ax = fig.get_ax(1, 1)
    ax.axis("off")

    return fig


def add_minimums_to_ax(fig: MyFig, array: np.ndarray, i: int, j: int, is_inverted: bool = False) -> MyFig:
    """
    :param fig: figure on which data is shown
    :param array: array of data to show
    :param i: row num of subplot
    :param j: column num of subplot
    :param is_inverted: True - if inverted, otherwise False
    :return: MyFig
    """

    ax = fig.get_ax(i, j)
    nums = local_minimum_numlist(array)
    list_of_values = [array[i] for i in nums]

    if is_inverted:
        ax.plot(list_of_values, nums, linestyle='', marker='.')
    else:
        ax.plot(nums, list_of_values, linestyle='', marker='.')

    return fig


def color_to_bw(img: np.ndarray, color_component: int = 0) -> np.ndarray:
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


def color_to_gray(img: np.ndarray) -> np.ndarray:
    img_gray = img.copy()

    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            (r, g, b) = img_gray[i, j, ::-1]
            gray = int((max(r, g, b) + min(r, g, b)) / 2)
            img_gray[i, j] = gray % 255

    return img_gray


def compute_statistics(projection: np.ndarray) -> dict:
    stats = {
        "Mean": np.mean(projection),
        "Median": np.median(projection),
        "Variance": np.var(projection),
        "Std Dev": np.std(projection),
        "Min": np.min(projection),
        "Max": np.max(projection),
        "Skewness": skew(projection),
        "Kurtosis": kurtosis(projection)
    }
    return stats


def vertical_proection(img: np.ndarray) -> np.ndarray:
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
        v_pr = np.zeros((arr1.size, 3))
        for i in range(arr1.size):
            v_pr[i, 0] = arr1[i]
            v_pr[i, 1] = arr2[i]
            v_pr[i, 2] = arr3[i]

    return v_pr


# produce matrix with rows HEIGHT (img.shape[1]) and cols (img.shape[2])
# if its 3 colored, then colors will be in (b, g, r)
def horizontal_proection(img: np.ndarray) -> np.ndarray:
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
        h_pr = np.zeros((arr1.size, 3))
        for i in range(arr1.size):
            h_pr[i, 0] = arr1[i]
            h_pr[i, 1] = arr2[i]
            h_pr[i, 2] = arr3[i]

    return h_pr


# Send ONLY one dimensional array
# Produce list of nums of minimum, that is less than average value
def local_minimum_list(img: np.ndarray) -> list:
    lminimums = list()
    for i in range(1, img.size):
        lminimums.append(img[i] - img[i - 1])

    average = max(lminimums) - min(lminimums)
    print("average_val: ", average)

    def lminimumsnum(average: float, lminimums: list) -> list:
        lminimumsnum = list()
        for i in range(len(lminimums)):
            if lminimums[i] <= average:
                lminimumsnum.append(i)

        return lminimumsnum

    return lminimumsnum(average, lminimums)


def local_minimum_numlist(array: np.ndarray) -> list:
    """
    :param array: one dimensional array of any projection
    :return: list with element nums of minimums
    """
    res = list()
    for i in range(1, array.shape[0] - 1):
        if array[i] < array[i - 1] and array[i] < array[i + 1]:
            res.append(i)

    return res


class MODE(Enum):
    TEST = 0
    RUN = 1


def client(mode: MODE, img_name: str) -> None:
    if mode == MODE.TEST:
        img = open_img(img_name)
    else:
        img = open_img()

    if img is None:
        return

    if IS_SHOW_ORIGINAL_IMG:
        cv2.imshow("Your image", img)
        cv2.waitKey(0)

    bw_img = color_to_bw(img, 2)
    gray_img = color_to_gray(img)

    cv2.imwrite("bw_img.png", bw_img)
    cv2.imwrite("gray_img.png", gray_img)

    if IS_SHOW_PRODUCED_IMG:
        cv2.imshow("BW_image (blue_component)", bw_img)
        cv2.imshow("gray_image", gray_img)
        cv2.waitKey(0)

    # Расчитать ряды данных вертикальных и горизонатльных проекций

    hp_res_for_gray = horizontal_proection(gray_img)

    vp_res_for_gray = vertical_proection(gray_img)

    if IS_SHOW_DISTINCT_GRAPHICS_FOR_PROJECTIONS:
        plt.plot(hp_res_for_gray[:, 0], linestyle='--', marker='.')
        plt.title("HP")
        nums = local_minimum_numlist(hp_res_for_gray[:, 0])
        vals = [hp_res_for_gray[i, 0] for i in nums]
        plt.plot(nums, vals, linestyle='', marker='.')
        plt.show()

        plt.plot(vp_res_for_gray[:, 0], linestyle='--', marker='.')
        nums = local_minimum_numlist(vp_res_for_gray[:, 0])
        vals = [vp_res_for_gray[i, 0] for i in nums]
        plt.plot(nums, vals, linestyle='', marker='.')
        plt.title("VP")
        plt.show()

    # Знаходження локальних мінімумов горизонтальної та вертикальної проекцій

    # use local_minimum_numlist

    # Отрисовка графиков проеций и изображения
    fig = MyFig(2, 2)

    unite_plot_new(fig, gray_img, hp_res_for_gray[:, 0], vp_res_for_gray[:, 0])
    add_minimums_to_ax(fig, hp_res_for_gray[:, 0], 0, 1, True)
    add_minimums_to_ax(fig, vp_res_for_gray[:, 0], 1, 0)

    fig.show()

    # Обчислення характериситик

    statistic_for_hp = compute_statistics(hp_res_for_gray[:, 0])
    statistic_for_vp = compute_statistics(vp_res_for_gray[:, 0])

    print("Infromation about statistic for horizontal proection:")
    for i in statistic_for_hp:
        print(f"{i}: {statistic_for_hp.get(i)}")

    print()
    print("Infromation about statistic for vertical proection:")
    for i in statistic_for_vp:
        print(f"{i}: {statistic_for_vp.get(i)}")


class Tests:
    @staticmethod
    def is_eq(m1, m2):
        for i in range(len(m1)):
            if m1[i] != m2[i]:
                return False

        return True

    @staticmethod
    def test_hp():
        matrix = np.array([
            [159, 223, 248, 15],
            [224, 118, 178, 192],
            [246, 8, 94, 66],
            [166, 66, 147, 209],
            [40, 94, 79, 137]
        ])
        res = np.array([645, 712, 414, 588, 350])
        t1 = horizontal_proection(matrix)
        assert Tests.is_eq(t1, res) == True
        print("hp Test passed")

    @staticmethod
    def test_img():
        img = open_img("cofee.png")
        print(f"Length: {img.shape[0]}; \n Width: {img.shape[1]}")

    @staticmethod
    def test_ndarray():
        pass

    @staticmethod
    def test_localminimum():
        testdata = np.array([15, 4, 3, 12, 15, 18, 4, 13])
        resdata = [3, 4]
        assert resdata == [testdata[i] for i in local_minimum_numlist(testdata)]


IS_SHOW_ORIGINAL_IMG = False
IS_SHOW_PRODUCED_IMG = False
IS_SHOW_DISTINCT_GRAPHICS_FOR_PROJECTIONS = False

if __name__ == "__main__":
    client(MODE.TEST, "cofee.png")
