from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
import cv2
import imutils


def show_image(imageStr, image):
    cv2.imshow(imageStr, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def return_cnts(image_c, keep):
    cnts = cv2.findContours(image_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if keep != None:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]
    return cnts


def draw_contours(image, cnts):
    for index, i in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(cnts[index])
        ar = w / float(h)
        print("Height", h, "width", w)
        print(ar)
        cv2.drawContours(image, cnts, index, (0, 0, 255), 2)
        print("hereeee", ar)
    show_image("Contours", image)


def divide_by_digits(image, cnts, cont):
    (x, y, w, h) = cv2.boundingRect(cnts)
    digit = image[y : y + h, x : x + w]
    show_image("Digit", digit)
    # cv2.imwrite("dataset/" + str(cont) + ".png", digit)


image = "images/alphabet.png"
image = cv2.imread(image)
show_image("Original", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

show_image("Light", light)

cnts = return_cnts(light, 70)
draw_contours(image, cnts)
cont = 0
for c in cnts:
    cont += 1
    print(cont)
    divide_by_digits(light, c, cont)
