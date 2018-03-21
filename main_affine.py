from PIL import Image
from SIFT1 import SIFT_
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import expm
import sys
from numpy.linalg import inv
import functools
from scipy.ndimage.interpolation import affine_transform

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

def RANSAC(im1,im2,f1,f2,th=5):
	print('RANSAC...')
	max_xy = [0,0]
	max_set = [[],[]]
	max_count = 0
	f1m = []
	f2m = []
	for f in f1:
		count=0
		xy_set = [[],[]]
		f1_ = []
		f2_ = []
		for  f_ in f1:
			if abs(f[-2]-f_[-2])+abs(f[-3]-f_[-3]) < 10:
				count += 1
				if [f_[0]-f_[-3],f_[1]-f[-2],1] not in xy_set[0] and [f_[0],f_[1],1] not in xy_set[1]:
					xy_set[0].append([f_[0],f_[1],1])
					xy_set[1].append([f_[0]+f_[-3],f_[1]+f_[-2],1])
					f1_.append(f_[:2])
					f2_.append([f_[0]+f_[-3],f_[1]+f_[-2]])

		if count >= max_count and f[-2]<=0:
			max_count = count
			max_xy = [f[-3],f[-2]]
			max_set = xy_set
			f1m = f1_
			f2m = f2_
	
	for ind,i in enumerate(max_set[0]):
		max_set[0][ind] = [i[0]+max_xy[0],i[1]+max_xy[1],1]

	visualization(im1,im2,f1m,f2m)
	print(max_count)
	print(max_set)
	return -max_xy[0], -max_xy[1], max_set

def get_KR(t):
	K = np.diag([t[0],t[0],1])
	t_matrix = np.array([[0,-t[3],t[2]],[t[3],0,-t[1]],[-t[2],t[1],0]])
	R = expm(t_matrix)
	return K,R

def update_mse(t1,t2,P1,P2,err0):
	K1,R1 = get_KR(t1)
	K2,R2 = get_KR(t2)

	err = 0
	for a,b in zip(P1,P2):
		proj = functools.reduce(np.dot,[K1,R1,R2.transpose(),inv(K2),b])
		err += (a[0]/a[2]-proj[0]/proj[2])**2+(a[1]/a[2]-proj[1]/proj[2])**2
	err /= len(P1)
	print('Avg err is:',err)

	#J will be 2n*8
	I = np.eye(8)
	n = len(P1)
	J = np.zeros([2*n,8])
	r = np.zeros([2*n,1])
	l = 0.001

	for ind,(a_,b_) in enumerate(zip(P1,P2)):
		p_ = functools.reduce(np.dot,[K1,R1,R2.transpose(),inv(K2),b_])
		r[2*ind:(ind+1)*2] = np.array([[a_[0]/a_[2]-p_[0]/p_[2]],[a_[1]/a_[2]-p_[1]/p_[2]]])
		J[2*ind:(ind+1)*2] = partialp(a_,b_,K1,R1,K2,R2)

	theta_ = inv(J.transpose().dot(J)+l*I).dot(J.transpose()).dot(r)

	t1 = [i+j[0] for i,j in zip(t1,theta_[0:4])]
	t2 = [i+j[0] for i,j in zip(t2,theta_[4:8])]

	
	return err,t1,t2

def partialp(a_,b_,K1,R1,K2,R2):
	pp_ = [[1/b_[2],0,-b_[0]/(b_[2]**2)],[0,1/b_[2],-b_[1]/(b_[2]**2)]]
	p_fb = np.array(functools.reduce(np.dot,[pp_,K1,R1,R2.transpose(),[[-1/(K2[0,0]**2),0,0],[0,-1/(K2[0,0]**2),0],[0,0,0]],b_]))
	p_fa = np.array(functools.reduce(np.dot,[pp_,[[1,0,0],[0,1,0],[0,0,0]],R1,R2.transpose(),inv(K2),b_]))
	p_ta1 = np.array(functools.reduce(np.dot,[pp_,K1,R1,[[0,0,0],[0,0,-1],[0,1,0]],R2.transpose(),inv(K2),b_]))
	p_ta2 = np.array(functools.reduce(np.dot,[pp_,K1,R1,[[0,0,1],[0,0,0],[-1,0,0]],R2.transpose(),inv(K2),b_]))
	p_ta3 = np.array(functools.reduce(np.dot,[pp_,K1,R1,[[0,-1,0],[1,0,0],[0,0,0]],R2.transpose(),inv(K2),b_]))
	p_tb1 = np.array(functools.reduce(np.dot,[pp_,K1,R1,R2.transpose(),[[0,0,0],[0,0,1],[0,-1,0]],inv(K2),b_]))
	p_tb2 = np.array(functools.reduce(np.dot,[pp_,K1,R1,R2.transpose(),[[0,0,-1],[0,0,0],[1,0,0]],inv(K2),b_]))
	p_tb3 = np.array(functools.reduce(np.dot,[pp_,K1,R1,R2.transpose(),[[0,1,0],[-1,0,0],[0,0,0]],inv(K2),b_]))
	return np.vstack([p_fa,p_ta1,p_ta2,p_ta3,p_fb,p_tb1,p_tb2,p_tb3]).transpose()

def bundle(t1,t2,P):
	err = 5000
	#for i in range(500):
	while err > 20:
		err,t1,t2 = update_mse(t1,t2,P[0],P[1],err)
	return t1,t2

def registration(im1,im2,f1,f2):
	print('registration processing...')
	if not Path('f1_.npy').exists():
		f1 = match_point(f1,f2)
		np.save('f1_.npy',f1)
	f1 = np.load('f1_.npy')
	x,y,pairs = RANSAC(im1,im2,f1,f2)
	return int(x),int(y),pairs

def get_image(im1,im2):
	img01 = Image.open(im1)
	img1 = img01.convert('L')
	img02 = Image.open(im2)
	img2 = img02.convert('L')
	print(np.array(img1).shape)
	return img01,img1,img02,img2

def handle_SIFT(img1,img2):
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

	return f1,f2

def visualization(im1,im2,f1,f2):
	print(f1,f2)
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

def main(im1='01.jpg', im2='02.jpg'):
	img01,img1,img02,img2 = get_image(im1,im2)				#get rgb, gray scale PIL object of 2 image 
	
	f1,f2 = handle_SIFT(img1,img2)							#get feature set 1, feature set 2

	x,y,matched_pairs = registration(img01,img02,f1,f2)

	t1,t2 = bundle([500,0,0,0],[500,0,0,0],matched_pairs)

	print(t1,t2)

	K1,R1 = get_KR(t1)
	K2,R2 = get_KR(t2)

	f = functools.reduce(np.dot,[K1,R1,R2.transpose(),inv(K2)])
	
	#p = f.dot(matched_pairs[1][0])
	#p = p/p[2]
	#u = matched_pairs[0][0]

	#print(p,u)
	#x,y = p[0]-u[0],p[1]-u[1]
	
	new_img = img02.transform(img02.size, Image.AFFINE, f.ravel().tolist())
	img02 = np.array(new_img)
	#new_img.show()

	
	row,col,_ = np.array(img01).shape

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
	im.save('u_result.jpg')

if __name__ == '__main__':
	if len(sys.argv) == 2:
		main(sys.argv[1],sys.argv[2])
	else:
		main()




