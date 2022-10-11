from turtle import width
import cv2

from apply_presepctive import apply_prespective

for i in range(1, 17):
    image = cv2.imread("images/cotxe{}.jpg".format(i))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gauss = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gauss, 100, 100)
    # cv2.imshow("imatge", edges), cv2.waitKey(0), cv2.destroyAllWindows()

    # cv2.imshow('i', edges), cv2.waitKey(0),cv2.destroyAllWindows()

    cnts, h = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # print("He encontrado {} objetos".format(len(cnts)))

    ll = []
    max_area = 0
    max_area_c = 0
    for index, c in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        if x != 0 or y != 0:
            area = cv2.contourArea(c)
            # area = w * h
            aspect_ratio = w / float(h)
            # cv2.drawContours(image, cnts, index, (0, 0, 255), 2)
            # print(index, ",", aspect_ratio)
            if 2.3 < aspect_ratio < 3.3:
                # print("bbbbbbbbbb")
                # print(area)
                # cv2.drawContours(image, cnts, index, (0, 0, 255), 2)
                # if 23000 < area < 25000:
                if area > max_area:
                    max_area = area
                    max_area_c = index
                    ll.append(area)
                    print("aaaaaaaaaa")
                    print(index)
                    # cv2.drawContours(image, cnts, index, (0, 0, 255), 2)

    print(len(ll))

    # (x, y, w, h) = cv2.boundingRect(cnts[418])
    # area = cv2.contourArea(cnts[418])
    # aspect_ratio = w / float(h)
    # print(aspect_ratio)
    # print(area)
    cv2.drawContours(image, cnts, max_area_c, (0, 0, 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    (x, y, w, h) = cv2.boundingRect(cnts[max_area_c])
    cropped = image[y : y + h, x : x + w]
    # cv2.imshow("Show Boxes", cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("crop-images/blobby" + str(i) + ".png", cropped)

    apply_prespective(cropped, x, y, w, h)
