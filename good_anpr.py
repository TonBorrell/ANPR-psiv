from pyexpat import model
import cv2
from matplotlib.artist import get
import numpy as np
import imutils
from skimage.segmentation import clear_border
import pytesseract as tess
import easyocr
import os
import joblib

from model import *


def show_image(imageStr, image):
    cv2.imshow(imageStr, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def return_cnts(image_c, keep):
    cnts = cv2.findContours(image_c.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if keep != None:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]
    return cnts


def locate_license_plate(gray, candidates, image_filename):
    lpCnt = None
    roi = None
    minAR = 1.4
    maxAR = 3.4
    for c in candidates:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if ar >= minAR and ar <= maxAR:
            lpCnt = c
            licensePlate = gray[y : y + h, x : x + w]
            roi = cv2.threshold(
                licensePlate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )[1]
            roi = clear_border(roi)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            warped = apply_prespective(roi, approx, image_filename)

            return (roi, lpCnt), warped


def apply_prespective(image, approx, image_filename):
    src = np.squeeze(approx).astype(np.float32)
    height = image.shape[0]
    width = image.shape[1]
    dst = np.float32([[0, 0], [0, height - 1], [width - 1, 0], [width - 1, height - 1]])
    src = order_points(src)
    dst = order_points(dst)

    M = cv2.getPerspectiveTransform(src, dst)
    img_shape = (width, height)
    # He cambiat aixo !!!!
    image_original = cv2.imread(image_filename)
    light = get_light_image(image_original)
    warped = cv2.warpPerspective(light, M, img_shape, flags=cv2.INTER_LINEAR)
    global original_warped
    original_warped = cv2.warpPerspective(
        image_original, M, img_shape, flags=cv2.INTER_LINEAR
    )
    roi = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    return roi


def order_points(pts):
    # Step 1: Find centre of object
    center = np.mean(pts)

    # Step 2: Move coordinate system to centre of object
    shifted = pts - center

    # Step #3: Find angles subtended from centroid to each corner point
    theta = np.arctan2(shifted[:, 0], shifted[:, 1])

    # Step #4: Return vertices ordered by theta
    ind = np.argsort(theta)
    return pts[ind]


def clean_unnecesary_cnts(image, cnts):
    borrar_list = []
    for index in range(len(cnts)):
        if cv2.contourArea(cnts[index]) > 11600:
            borrar_list.append(index)
        if cv2.contourArea(cnts[index]) < 1000:
            borrar_list.append(index)

    for i in reversed(borrar_list):
        cnts.pop(i)

    index_borrar = []
    for index, cnt in enumerate(cnts):
        for i in cnt:
            if i[0][1] == 0:
                index_borrar.append(index)
            if i[0][0] == 0:
                index_borrar.append(index)

    index_borrar = list(set(index_borrar))
    for i in reversed(index_borrar):
        cnts.pop(i)

    return cnts


def divide_by_letters(image, type="numbers"):
    cnts = return_cnts(image, 9)
    cnts = clean_unnecesary_cnts(image, cnts)

    cnts_list = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cnts_list.append(x)

    cnts_list = sorted(cnts_list)
    try:
        number_letter_separation = (cnts_list[3] + cnts_list[4]) // 2 + 20

        y, x = image.shape

        numbers = image[0:y, 0:number_letter_separation]
        letters = image[0:y, number_letter_separation:x]

        # draw_contours(image, cnts, 0, 100)
        if type == "numbers":
            return numbers
        else:
            return letters
    except:
        return image


def return_digits(image, cnts):
    (x, y, w, h) = cv2.boundingRect(cnts)
    digit = image[y : y + h, x : x + w]

    text = tess.image_to_string(digit, lang="eng", config="--psm 10")
    predicted_result = tess.image_to_string(
        digit,
        lang="eng",
        config="--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789BCDFGHJKLMNPQRSTVWXYZ",
    )

    """
    reader = easyocr.Reader(["en"])
    result = reader.readtext(image)

    """

    # print("Text", result)

    show_image("Digit", digit)

    return digit


def draw_contours(image, cnts, minAR=None, maxAR=None):
    for index, i in enumerate(cnts):
        if maxAR and minAR:
            (x, y, w, h) = cv2.boundingRect(cnts[index])
            ar = w / float(h)
            if ar >= minAR and ar <= maxAR:
                cv2.drawContours(image, cnts, index, (0, 0, 255), 2)
        else:
            cv2.drawContours(original_warped, cnts, index, (0, 0, 255), 2)
            show_image("Contours", original_warped)
    show_image("Contours", image)


def draw_all_contours(image, cnts):
    for index, c in enumerate(cnts):
        cv2.drawContours(image, cnts, index, (0, 0, 255), 2)
        show_image("Contours", image)


def get_plate_digits_from_images(digit_type, folder):
    digit_return_images = []
    all_files = os.listdir(folder)
    for file in all_files:
        print("Filename", folder + file)
        image = cv2.imread(folder + file)
        light = get_light_image(image)
        # show_image("Test", light)
        cnts = return_cnts(light, 10)
        (_, _), warped = locate_license_plate(light, cnts, folder + file)
        cnts = return_cnts(warped, 9)
        cnts = clean_unnecesary_cnts(warped, cnts)
        cnts_list = []
        for c in cnts:
            (x, y, _, _) = cv2.boundingRect(c)
            cnts_list.append(x)

        cnts_list = sorted(cnts_list)

        try:
            number_letter_separation = (cnts_list[3] + cnts_list[4]) // 2 + 20

            y, x = warped.shape

            if digit_type == "number":
                numbers = warped[0:y, 0:number_letter_separation]
                digit_return_images.append((numbers))
                # return numbers
            elif digit_type == "letter":
                letters = warped[0:y, number_letter_separation:x]
                digit_return_images.append((letters))
                # return letters
        except:
            pass
            # return image
    return digit_return_images


def set_size(image, x, y):
    dim = (x, y)
    image_resize = cv2.resize(image, dim)
    return image_resize


cont = 0
cont_numbers = 0


def get_one_number(image):
    global cont, cont_numbers
    cont += 1
    cnts = return_cnts(image, 6)
    # draw_all_contours(image, cnts)
    numbers = []
    cnts = clean_unnecesary_cnts(image, cnts)
    for index, c in enumerate(cnts):
        cont_numbers += 1
        (x, y, w, h) = cv2.boundingRect(c)
        digit = image[y : y + h, x : x + w]
        # show_image("Digit", digit)
        # cv2.imwrite("dataset/real/" + str(cont) + "+" + str(index) + ".jpg", digit)

        numbers.append((digit, x))

    numbers.sort(key=lambda x: x[1])
    number_end = []
    for (number, x) in numbers:
        number_end.append(number)

    return number_end


def get_light_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
    light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    return light


def predict(folder, image_filename):
    numbers_real = image_filename.split("_")[0]
    letters_real = image_filename.split("_")[1]
    letters_real = letters_real.split(".")[0]

    image_filename = folder + image_filename
    image = cv2.imread(image_filename)
    light = get_light_image(image)
    cnts = return_cnts(light, 10)
    draw_contours(light, cnts, 1.4, 3.3)
    (roi, lpCnt), warped = locate_license_plate(light, cnts, image_filename)
    number = divide_by_letters(warped, "numbers")
    letter = divide_by_letters(warped, "letters")
    model_numbers_knn = joblib.load("model/knn/model_numbers")
    model_numbers_mlp = joblib.load("model/mlp/model_numbers")
    model_numbers_svm = joblib.load("model/svm/model_numbers")
    model_letters_knn = joblib.load("model/knn/model_letters")
    model_letters_mlp = joblib.load("model/mlp/model_letters")
    model_letters_svm = joblib.load("model/svm/model_letters")
    digits_numbers = get_one_number(number)
    numbers_pred_knn = ""
    numbers_pred_mlp = ""
    numbers_pred_svm = ""
    letters_pred_knn = ""
    letters_pred_mlp = ""
    letters_pred_svm = ""
    for digit in digits_numbers:
        digit = set_size(digit, 66, 240)
        digit = digit.reshape((1, 240 * 66))
        numbers_pred_knn += model_numbers_knn.predict(digit)[0]
        numbers_pred_mlp += model_numbers_mlp.predict(digit)[0]
        numbers_pred_svm += model_numbers_svm.predict(digit)[0]

    digits_letters = get_one_number(letter)
    for digit in digits_letters:
        digit = set_size(digit, 66, 240)
        digit = digit.reshape((1, 240 * 66))
        letters_pred_knn += model_letters_knn.predict(digit)[0]
        letters_pred_mlp += model_letters_mlp.predict(digit)[0]
        letters_pred_svm += model_letters_svm.predict(digit)[0]

    print("Matricula a predir", numbers_real, letters_real)
    print("Matricula predida KNN", numbers_pred_knn, letters_pred_knn)
    print("Matricula predida mlp", numbers_pred_mlp, letters_pred_mlp)
    print("Matricula predida svm", numbers_pred_svm, letters_pred_svm)

    if numbers_real == numbers_pred_knn and letters_real == letters_pred_knn:
        return 1
    else:
        show_contrours_image(image, image_filename, digits_numbers, digits_letters)
        return 0


def show_contrours_image(image, image_filename, digit_numbers, digit_letters):
    light = get_light_image(image)
    cnts = return_cnts(light, 10)
    for c in range(len(cnts)):
        cv2.drawContours(image, cnts, c, (0, 0, 255), 2)
    _, warped = locate_license_plate(light, cnts, image_filename)
    cnts = return_cnts(warped, 10)
    for c in range(len(cnts)):
        cv2.drawContours(original_warped, cnts, c, (0, 0, 255), 2)

    img_read1 = cv2.resize(image, (250, 250))
    img_read2 = cv2.resize(original_warped, (250, 250))
    col1 = np.hstack([img_read1, img_read2])
    for d in digit_numbers:
        d = cv2.cvtColor(d, cv2.COLOR_GRAY2RGB)
        img_read1 = cv2.resize(d, (250, 250))
        col1 = np.hstack([col1, img_read1])
    for d in digit_letters:
        d = cv2.cvtColor(d, cv2.COLOR_GRAY2RGB)
        img_read1 = cv2.resize(d, (250, 250))
        col1 = np.hstack([col1, img_read1])

    show_image("Error", col1)


images = os.listdir("images/good/")
folder = "images/good/"
cont_total = 0
cont_true = 0
for image in images:
    if ".jpg" in image:
        print(image)
        ok = predict(folder, image)
        cont_total += 1
        cont_true += ok

print("Accuracy", cont_true / cont_total)
