import cv2
from apply_presepctive import apply_prespective
import numpy as np

i = 16
image = cv2.imread("images/cotxe{}.jpg".format(i))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gauss = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(gauss, 100, 100)
# cv2.imshow("imatge", edges), cv2.waitKey(0), cv2.destroyAllWindows()

# cv2.imshow('i', edges), cv2.waitKey(0),cv2.destroyAllWindows()

cnts, h = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print("He encontrado {} objetos".format(len(cnts)))

ll = []
max_area = 0
max_area_c = 0
max_area_contour = None
for index, c in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    if x != 0 or y != 0:
        area = cv2.contourArea(c)
        aspect_ratio = w / float(h)
        # print(index, ",", aspect_ratio)
        if 2 < aspect_ratio < 3.3:
            if area > max_area:
                max_area = area
                max_area_c = index
                max_area_contour = c
                ll.append(area)

lst_intensities = []
cv2.drawContours(image, cnts, max_area_c, (0, 0, 255), 2)
pts = np.where(image == 255)
lst_intensities.append(image[pts[0], pts[1]])
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("########################")
print(cnts[max_area_c])
print('""""""""""""""')
(x, y, w, h) = cv2.boundingRect(cnts[max_area_c])
print(x, y)

point_list = []
min_bot_left = 10000
max_bot_right = 0
for index, i in enumerate(cnts[max_area_c]):
    if index % 50 == 0:
        for (xi, yi) in i:
            image = cv2.circle(
                image, (xi, yi), radius=5, color=(0, 255, 255), thickness=-1
            )
            if xi == x:
                min_bot_left = yi
                print("aaaaaaa")

print("Height", h, "width", w)
area = cv2.contourArea(max_area_contour)
print("Area", area, "Area2", w * h)

image_cropped = image[y : y + h, x : x + w]
image_cropped = cv2.circle(
    image_cropped, (x, y), radius=5, color=(0, 255, 255), thickness=-1
)
print(cnts[max_area_c][-1][0][0])
image = cv2.circle(
    image,
    (cnts[max_area_c][-1][0][0], cnts[max_area_c][-1][0][1]),
    radius=5,
    color=(255, 255, 255),
    thickness=-1,
)
cv2.imshow("Image", image_cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
apply_prespective(image, x, y, w, h)
