import math
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
original_path = 'grigio_piombo_patch/patchii2/abnormal/striscifini'
destination_path = 'grigio_piombo_hougeslinesp'

format_of_your_images = 'jpg'
format_of_your_images2 = 'png'

all_the_files_jpg = Path(original_path).rglob(f'*.{format_of_your_images}')
all_the_files_png = Path(original_path).rglob(f'*.{format_of_your_images2}')
a=0
print(all_the_files_jpg)
for f in all_the_files_jpg:
    print(a)
    a=a+1
    p = cv2.imread(str(f),0)
    out=cv2.cvtColor(p,cv2.COLOR_GRAY2BGR)
    gaussian = cv2.GaussianBlur(p, (555, 555), 0)
    gaussian = 255 - (p - gaussian)
    kernel = np.ones((5, 5), np.uint8)
    gaussian = cv2.morphologyEx(gaussian, cv2.MORPH_OPEN, kernel)
    ret2,gaussian = cv2.threshold(gaussian,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    gaussian = cv2.morphologyEx(gaussian, cv2.MORPH_OPEN, kernel)

    # gaussian[gaussian>100]=255
    # gaussian[gaussian <= 100] = 0


    # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    # cv2.imshow("test", gaussian)
    # cv2.waitKey(1)
    # gaussian = cv2.Canny(gaussian, 20, 255, L2gradient=True)
    # gaussian = cv2.medianBlur(p, 75)

    # gaussian[gaussian < 0.5 * np.max(gaussian)]=0
    #  transformation
    # Detect and draw lines
    lines = cv2.HoughLinesP(gaussian, 1, np.pi / 360, 200, minLineLength=256, maxLineGap=30)
    if lines is not None:
        for line in lines [:1]:
            for x1, y1, x2, y2 in line:
                cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

    gaussian = cv2.cvtColor(gaussian, cv2.COLOR_RGB2BGR)
    cv2.imshow('out', np.hstack([gaussian, out]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(f'{destination_path}/{f.name}', gaussian)



for f in all_the_files_png:
    print(a)
    a = a + 1
    p = cv2.imread(str(f), 0)
    out = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
    gaussian = cv2.GaussianBlur(p, (555, 555), 0)
    gaussian = 255 - (p - gaussian)
    kernel = np.ones((5, 5), np.uint8)
    gaussian = cv2.morphologyEx(gaussian, cv2.MORPH_OPEN, kernel)
    ret2, gaussian = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    gaussian = cv2.morphologyEx(gaussian, cv2.MORPH_OPEN, kernel)

    # gaussian[gaussian>100]=255
    # gaussian[gaussian <= 100] = 0

    # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    # cv2.imshow("test", gaussian)
    # cv2.waitKey(1)
    # gaussian = cv2.Canny(gaussian, 20, 255, L2gradient=True)
    # gaussian = cv2.medianBlur(p, 75)

    # gaussian[gaussian < 0.5 * np.max(gaussian)]=0
    #  transformation
    # Detect and draw lines
    lines = cv2.HoughLinesP(gaussian, 1, np.pi / 360, 200, minLineLength=256, maxLineGap=30)
    if lines is not None:
        for line in lines[:1]:
            for x1, y1, x2, y2 in line:
                cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

    gaussian = cv2.cvtColor(gaussian, cv2.COLOR_RGB2BGR)
    cv2.imshow('out', np.hstack([gaussian, out]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(f'{destination_path}/{f.name}', gaussian)




# sobelx = cv2.Sobel(gaussian,cv2.CV_64F,1,0,ksize=5)

#ret,thresh1 = cv2.threshold(gaussian,50,255,cv2.THRESH_BINARY)
#ret,thresh3 = cv2.threshold(gaussian,120,255,cv2.THRESH_TRUNC)

# kernel = np.ones((3,3),np.uint8)

# clean1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, np.ones((3, 3)))
# clean3 = cv2.morphologyEx(thresh3, cv2.MORPH_OPEN, np.ones((3, 3)))
#
# closing1 = cv2.morphologyEx(clean1, cv2.MORPH_CLOSE, kernel)
# closing3 = cv2.morphologyEx(clean3, cv2.MORPH_CLOSE, kernel)
#
# dilation1 = cv2.dilate(closing1,kernel,iterations = 2)
# dilation3 = cv2.dilate(closing3,kernel,iterations = 2)
#
# # gradient1 = cv2.morphologyEx(dilation1, cv2.MORPH_GRADIENT, kernel)
# # gradient3 = cv2.morphologyEx(dilation3, cv2.MORPH_GRADIENT, kernel)
