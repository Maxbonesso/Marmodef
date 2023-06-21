import cv2
import matplotlib.pyplot as plt
import numpy as np

# Carico immagine a
a = cv2.imread('Es1/a.jpg')
# Grayscale a
Ia = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

# Carico immagine b
b = cv2.imread('Es1/b.jpg')
# Grayscale b
Ib = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

# histogram of a
histogram = cv2.calcHist([Ia], [0], None, [256], [0, 256])
plt.plot(histogram, color='k')
plt.show()

# applying Otsu thresholding to a
ret, Ibw = cv2.threshold(Ia, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# saving output Otsu thresholding image
result = 'abw.png'
cv2.imwrite(result, Ibw)

# create an image based on dimension of Ib
# h, w = Ib.shape
# Imix = 255 * np.ones(shape=(h, w), dtype=np.uint8)
# create the final image
# for x in range(h) :
#    for y in range(w) :
#        if Ibw[x][y]==0 :
#            Imix[x][y]=Ib[x][y]
#        else :
#            Imix[x][y] = Ia[x][y]

# create an image with same dimensions as input images
Imix = np.zeros_like(Ib, dtype=np.uint8)

# compare Otsu thresholding image and find zeros
zeros_mask = Ibw == 0

# where 'zeros_mask' is True, take Ib
Imix[zeros_mask] = Ib[zeros_mask]

# negated 'zeros_mask', take Ia
Imix[~zeros_mask] = Ia[~zeros_mask]

mix = 'mix.jpg'
cv2.imwrite(mix, Imix)
cv2.imshow('mix', Imix)
cv2.waitKey(0)
