import math
import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread('terraebano/terraebano_topatch/patchii2/MACRO_23_ii2.png')
x=0
y=512
x1=0
y1=512
name=0
maxX=4112
maxY=3008

while x < maxX and y < maxY:
    imgnew=img[x:y,x1:y1]
    cv2.imwrite('terraebano/terraebano_patch/patchii2/'+str(x)+"-"+str(y)+"-"+str(x1)+"-"+str(y1)+'MACRO_23_ii2.png', imgnew)
    name=name+1
    x1=x1+512
    y1=y1+512
    if x1>maxX or y1>maxX:
        x1=0
        y1=512
        x=x+512
        y=y+512