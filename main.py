from PIL import Image
from SIFT1 import SIFT_
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def match_point(f1,f2):
	print('Matching points...')
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

def RANSAC(col,f1,f2,th=5):
	print('RANSAC...')
	max_xy = [0,0]
	max_set = [[],[]]
	max_count = 0
	for f in f1:
		#print(f[-1])
		count=0
		xy_set = [[],[]]
		# this is for purdue
		for f_ in f2:
			if ((f_[0]-(f[0]+f[-3]))**2+(f_[1]-(f[1]+f[-2]))**2)**0.5 < 10:
				cos_ = np.dot(f[2:-3],f_[2:])/np.linalg.norm(f[2:-3],2)/np.linalg.norm(f_[2:],2)
				if cos_>0.7:
					#if f[1]>150 and f_[1]<150:
					count += 1
		
		'''----------------------real image 5,6
		for f_ in f2:
			#if f[-3]==0 and f[-2]==-252:
			#	print('p')
			#if f_[1]-f[1] == -252 and f_[0]-f[0]==0:
			#	print(cosine_similarity([f[2:-3]],[f_[2:]]))
			if ((f_[0]-(f[0]+f[-3]))**2+(f_[1]-(f[1]+f[-2]))**2)**0.5 < 35:
				#if f_[1]-f[1] == -252 and f_[0]-f[0]==0:
				#	print('here')
				#print (cosine_similarity([f[2:-3]],[f_[2:]])[0][0]>0.7)
				cos_ = np.dot(f[2:-3],f_[2:])/np.linalg.norm(f[2:-3],2)/np.linalg.norm(f_[2:],2)
				if cos_>0.7:
					#if f[-2]<-100:
					count += 1
					xy_set[0].append(f_[0]-f[0])
					xy_set[1].append(f_[1]-f[1])
		'''
		if count >= max_count and f[-2]<=0:
			max_count = count
			max_xy = [f[-3],f[-2]]
			max_set = xy_set
	print(max_count)
	#return -max_xy[0],-max_xy[1]
	print(max_set)
	return -int(sum(max_set[0])/len(max_set[0])),-int(sum(max_set[1])/len(max_set[1]))

def registration(col,f1,f2):
	print('registration processing...')
	if not Path('f1_.npy').exists():
		f1 = match_point(f1,f2)
		np.save('f1_.npy',f1)
	f1 = np.load('f1_.npy')
	x,y = RANSAC(col,f1,f2)
	print(x,y)
	return int(x),int(y)

def test():
	x1 = [[2,3,[1,2,3,4,5]],[2,3,[1,2,3,4,5]]]
	x2 = x1
	registration(np.zeros((3,3)),x1,x2)

def main():
	img01 = Image.open('images/05.jpg')
	print(img01)
	img1 = img01.convert('L')
	img02 = Image.open('images/06.jpg')
	img2 = img02.convert('L')
	
	print(np.array(img1).shape)

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
	print(np.array(img01).shape)
	row,col,_ = np.array(img01).shape
	x,y = registration(col,f1,f2)

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
	if 0:
		new_img = np.zeros([row,col+y,3])
		new_img[:,:col,:] = img01
		new_img[:,-col:,:] += img02
		new_img[:,y:col,:] /=2
	
	im = Image.fromarray(new_img.astype('uint8'))
	im.show()
	im.save('u_result.jpg')

if __name__ == '__main__':
	main()
	#test()




