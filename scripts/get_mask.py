# given an image with a mask of color 0 255 0 superimposed on top, return the mask
import sys
from PIL import Image
import numpy as np
sys.path.append("./")

mask_color = np.array([0, 255, 0, 255])
filename = "test.png"

def col_eq(a, b):
    for i, c in enumerate(a):
        if a[i] != b[i]: return False
    return True

img = np.array(Image.open(filename))
b = np.zeros((256, 256))
for i in range(len(img)):
    for j in range(len(img[0])):
        if col_eq(img[i][j], mask_color):
            b[i][j] = 1
temp = img.transpose(2, 0, 1) * b
# print(img.shape, img.transpose().shape)
temp = temp.transpose(1, 2, 0)
# print(temp[1][1][4])
temp = Image.fromarray(temp)
temp.save("masked.png")

# import pickle
# with open("test_mask.pickle", "wb") as f:
#     pickle.dump(b, f)
