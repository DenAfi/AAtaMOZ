import cv2
import numpy as np
import matplotlib.pyplot as plt

def reduce_intensity_levels(img, levels):
    step = 256 // levels
    return (img // step) * step

def average_filter(img, ksize):
    return cv2.blur(img, (ksize, ksize))

def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated

def block_average(img, block_size):
    h, w = img.shape[:2]
    img_out = img.copy()
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = img[y:y+block_size, x:x+block_size]
            mean = np.mean(block, axis=(0,1), dtype=int)
            img_out[y:y+block_size, x:x+block_size] = mean
    return img_out

def show(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    path = input("Введіть шлях до зображення: ")
    levels = int(input("Введіть бажану кількість рівнів інтенсивності (2, 4, 8, 16...): "))

    img = cv2.imread(path)
    if img is None:
        print("Помилка: не вдалося завантажити зображення.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Зменшення кількості рівнів інтенсивності
    reduced = reduce_intensity_levels(gray, levels)
    show("Зменшена кількість рівнів інтенсивності", reduced)

    # 2. Просте згладжування 3x3, 10x10, 20x20
    show("Усереднення 3x3", average_filter(gray, 3))
    show("Усереднення 10x10", average_filter(gray, 10))
    show("Усереднення 20x20", average_filter(gray, 20))

    # 3. Повороти
    show("Поворот на 45 градусів", rotate_image(gray, 45))
    show("Поворот на 90 градусів", rotate_image(gray, 90))

    # 4. Просторове зменшення роздільності
    show("Зменшення роздільності 3x3", block_average(gray, 3))
    show("Зменшення роздільності 5x5", block_average(gray, 5))
    show("Зменшення роздільності 7x7", block_average(gray, 7))

if __name__ == "__main__":
    main()
