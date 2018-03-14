from PIL import Image
from SIFT1 import SIFT_
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from pathlib import Path

im = np.array(Image.open('imm.jpg'))

row,col,cha = im.shape

im1 = im[:,int(col*2/5):]
im2 = im[:,:int(col*3/5)]

print (im1.shape,im2.shape)

result = Image.fromarray(im1)
result.save('imm2.jpg')

result = Image.fromarray(im2)
result.save('imm1.jpg')