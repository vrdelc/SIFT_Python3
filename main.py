from PIL import Image
from SIFT1 import SIFT_
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def match_point(f1,f2):
	f_comb = []
	for f in f1:				#f = feature
		max_cos = 0
		x,y = 0,0
		for f_ in f2:
			#cos = cosine_similarity([f[2:]],[f_[2:]])
			cos = np.dot(f[2:],f_[2:])/np.linalg.norm(f[2:],2)/np.linalg.norm(f_[2:],2)
			if cos > max_cos:
				max_cos = cos
				x,y = f_[0],f_[1]
		#print(max_cos)
		f_comb.append([i for i in f]+[x-f[0],y-f[1],max_cos])
	return f_comb

def RANSAC(im1,im2,f1,f2,th=5):
	max_xy = [0,0]
	max_count = 0
	f1m = []
	f2m = []
	for f in f1:
		#print(f[-1])
		count=0
		f1_ = []
		f2_ = []
		for  f__ in f1:
			for f_ in f2:
				if ((f_[0]-(f__[0]+f[-3]))**2+(f_[1]-(f__[1]+f[-2]))**2)**0.5 < 2:
					cos_ = np.dot(f__[2:-3],f_[2:])/np.linalg.norm(f__[2:-3],2)/np.linalg.norm(f_[2:],2)
					if cos_>0.8:
						count += 1
						f1_.append(f__[:2])
						f2_.append(f_[:2])

		if count >= max_count and f[-2]<=0:
			max_count = count
			max_xy = [f[-3],f[-2]]
			f1m = f1_
			f2m = f2_

	visualization(im1,im2,f1m,f2m)
	print(max_count)
	return -max_xy[0],-max_xy[1]

def registration(im1,im2,f1,f2):
	print('registration processing...')
	print('Matching points...')
	if not Path('f1_.npy').exists():
		f1 = match_point(f1,f2)
		np.save('f1_.npy',f1)
	f1 = np.load('f1_.npy')
	print('RANSAC...')
	if not Path('match.npy').exists():
		x,y = RANSAC(im1,im2,f1,f2)
		np.save('match.npy',[x,y])
	xy = np.load('match.npy')
	print(xy)
	return int(xy[0]),int(xy[1])

def visualization(im1,im2,f1,f2):
	plt.figure()
	plt.imshow(im1)
	for f in f1:
		plt.plot(f[1], f[0],marker='o', markerfacecolor='none',markeredgecolor='r')
	plt.show()
	plt.figure()
	plt.imshow(im2)
	for f in f2:
		plt.plot(f[1], f[0],marker='o', markerfacecolor='none',markeredgecolor='r')
	plt.show()

def test():
	x1 = [[2,3,[1,2,3,4,5]],[2,3,[1,2,3,4,5]]]
	x2 = x1
	registration(np.zeros((3,3)),x1,x2)

def main(im1='purdue/im1.jpg', im2='purdue/im2.jpg'):
	img01 = Image.open(im1)
	img1 = img01.convert('L')
	img02 = Image.open(im2)
	img2 = img02.convert('L')
	
	print('Shape of image1:',np.array(img1).shape)

	if not Path('f1.npy').exists():
		s1 = SIFT_(np.array(img1))
		f1 = s1.get_features().get_descriptor()
		print(f1[2],len(f1[2]))
		s2 = SIFT_(np.array(img2))
		f2 = s2.get_features().get_descriptor()
		np.save('f1.npy',f1)
		np.save('f2.npy',f2)
	
	f1 = np.load('f1.npy')
	f2 = np.load('f2.npy')
	#print([i[:2] for i in f1])
	row,col,_ = np.array(img01).shape
	x,y = registration(img01,img02,f1,f2)

	if x >= 0:
		new_img = np.zeros([row+x,col+y,3])
		new_img[:row,:col,:] = img01
		new_img[-row:,-col:,:] += img02
		new_img[x:row,y:col,:] /=2
	else:
		new_img = np.zeros([row-x,col+y,3])
		new_img[-row:,:col,:] = img01
		new_img[:row,-col:,:] += img02
		new_img[-x:row,y:col,:] /=2
	
	im = Image.fromarray(new_img.astype('uint8'))
	im.show()
	im.save('result.jpg')

if __name__ == '__main__':
	if len(sys.argv) == 2:
		main(sys.argv[1],sys.argv[2])
	else:
		main()




