import cv2

# Carico immagine bacteria
bacteria = cv2.imread('bacteria.png')
# Grayscale bacteria
graybacteria = cv2.cvtColor(bacteria, cv2.COLOR_BGR2GRAY)
# applying Otsu thresholding to bacteria
ret2, thresholdbacteria = cv2.threshold(graybacteria, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# detect the contours
result = bacteria.copy()
contours, hierarchy = cv2.findContours(thresholdbacteria, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#count the contours
cnt=len(contours)
#print the contours to the copy image
for x in range(cnt):
    if 200 >= cv2.arcLength(contours[x], True) and cv2.arcLength(contours[x], True) >= 100:
        cv2.drawContours(result, contours, x, (0,0,255), 1)

result_bacteria = 'result_bacteria.jpg'
cv2.imwrite(result_bacteria, result)
cv2.imshow('result_bacteria', result)
cv2.waitKey(0)