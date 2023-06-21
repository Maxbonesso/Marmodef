import math
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('000.png')
# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# testcanny = cv2.Canny(img,50,255, L2gradient=True)
#
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(testcanny,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()

# newimg = gray - cv2.medianBlur(gray, 25)
# thresh = cv2.threshold(newimg, 150, 255, cv2.THRESH_BINARY)[1]
# cv2.namedWindow("asd", cv2.WINDOW_NORMAL)
# cv2.imshow("asd", thresh)
# cv2.waitKey(0)
# adaptive threshold
# thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 33, 2)

thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]

# apply morphology
kernel = np.ones((3, 3), np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
# morph=cv2.dilate(thresh, kernel)
kernel = np.ones((3, 3), np.uint8)
# morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
# morph = cv2.medianBlur(morph, 5)

# cv2.namedWindow("prova", cv2.WINDOW_NORMAL)
# cv2.imshow("prova", morph)
# cv2.waitKey(0)
# exit()

# get hough line segments
threshold = 25
minLineLength = 10
maxLineGap = 20
lines = cv2.HoughLines(morph, 1, np.pi / 180, threshold)
if lines is None:
    lines = []
# draw lines
linear1 = np.zeros_like(thresh)
linear2 = img.copy()
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(linear2, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

print('number of lines:', len(lines))

# save resulting masked image
cv2.imwrite('scratches_thresh.jpg', thresh)
cv2.imwrite('scratches_morph.jpg', morph)
cv2.imwrite('scratches_lines1.jpg', linear1)
cv2.imwrite('scratches_lines2.jpg', linear2)
