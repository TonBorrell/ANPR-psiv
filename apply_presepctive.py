import cv2
import numpy as np
#import test


def apply_prespective(image, x, y, width, height):
    width_wanted = 330
    height_wanted = 110

    top_left = [x, y]
    top_right = [x + width, y]
    bot_left = [x, y + height]
    bot_right = [x + width, y + height]

    input = np.float32([top_left, top_right, bot_right, bot_left])
    output = np.float32(
        [
            [0, 0],
            [width_wanted, 0],
            [width_wanted, height_wanted],
            [0, height_wanted],
        ]
    )

    image_circle = cv2.circle(
        image, (x, y), radius=5, color=(0, 255, 255), thickness=-1
    )
    image_circle = cv2.circle(
        image, (x + width, y), radius=5, color=(255, 255, 255), thickness=-1
    )
    image_circle = cv2.circle(
        image, (x, y + height), radius=5, color=(0, 255, 255), thickness=-1
    )
    image_circle = cv2.circle(
        image, (x + width, y + height), radius=5, color=(0, 255, 255), thickness=-1
    )

    matrix = cv2.getPerspectiveTransform(input, output)

    print(matrix.shape)
    print(matrix)

    imgOutput = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    print(imgOutput.shape)

    # save the warped output
    # cv2.imwrite("sudoku_warped.jpg", imgOutput)

    # show the result
    cv2.imshow("result", imgOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
