import numpy as np
from PIL import Image
from pathlib import Path
import math
from scipy.signal import convolve2d
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
from numpy import linalg as LA

def get_image(file_name):
	file = Path('img_data.npy')
	if file.exists():
		data = np.load('img_data.npy')
		#print ('File exists.')
	else:
		im = open(file_name)
		lines = im.readlines()
		width,height = [int(i) for i in lines[2].split(' ')]
		raw_data = lines[4:]
		data = np.array([int(i.strip('\n')) for i in raw_data]).reshape((height,width))
		np.save('img_data.npy',data)
		#print (data)
	return data

def overflow(im):
	for i in np.arange(im.shape[0]):
		for j in np.arange(im.shape[1]):
			if im[i,j]>255:
				im[i,j] = 255
	return im

def kernel(sigma):
	k_size  = 4*int(sigma)+1
	k_width = 2*int(sigma)
	x, y = np.meshgrid(np.linspace(-k_width, k_width, k_size), np.linspace(-k_width, k_width, k_size))
	d = x*x+y*y
	return np.exp(-(d/(2.0*sigma**2)))/(2*math.pi*sigma**2)

def Gaussian_filter(im,sigma):
	#print ('Gaussian')
	g = kernel(sigma)
	return overflow(convolve2d(im, g,mode='same',boundary='symm')/sum(sum(g)))

class SIFT_:
	def __init__(self,img):
		self.origin_img = img
		self.img = np.array(Image.fromarray(img.astype('uint8'),'L').resize(np.multiply(2,[img.shape[1],img.shape[0]]),resample=Image.BICUBIC))
		#self.img = np.array(Image.fromarray(img.astype('uint8'),'L'))
		self.octaves = 3
		self.scales = 2
		self.factor = 2**0.2
		self.prior = 1.6
		self.ratio = 10
		self.dog = []
		self.orientation_mask_ratio = 1.5
		self.set_size = 4
		self.descriptor_size = 4
		self.orientation_bin_number = 10

	def get_features(self):
		self.DoG()
		self.Extrema()
		return self

	def get_descriptor(self):
		self.extrema = self.Orientation()
		self.mag,self.ori = self.assign_orientation_all()
		self.descriptor()
		return self.descriptor_

	def load_DoG(self):
		file = Path('files/octave3.npy')
		self.dog = []
		if file.exists():
			##print('Files exist, loading...')
			for i in range(4):
				self.dog.append(np.load('files/octave'+str(i)+'.npy'))
		else:
			##print('Files don\'t exist, processing DoG...')
			self.DoG()

	def load_Extrema(self):
		file = Path('files/octave3_2extrema.npy')
		self.extrema = []
		if file.exists():
			##print('Files exist, loading...')
			for i in range(4):
				self.extrema.append([])
				for j in range(3):
					self.extrema[i].append(np.load('files/octave'+str(i)+'_'+str(j)+'extrema.npy'))
		else:
			##print('Files don\'t exist, processing extrema...')
			self.Extrema()

	def DoG(self):
		self.octaves_ = []
		self.levels = self.scales+4
		for octave in range(self.octaves):
			self.octaves_.append([Gaussian_filter(self.img, self.prior*(self.factor**level)) for level in range(self.levels)])
			self.img = self.img[::2,::2]

		for ind,imgs in enumerate(self.octaves_):
			tmp = [np.subtract(imgs[ind+1],img) for ind,img in enumerate(imgs) if ind != len(imgs)-1]
			tmp = np.stack(tmp,axis=0)
			##print(tmp.shape)
			np.save('files/octave'+str(ind)+'.npy',tmp)
			self.dog.append(tmp)

	def Extrema(self):
		img = self.origin_img
		##print('Processing extrema')
		threshold = (self.ratio + 1) ** 2 / self.ratio 		#hessian
		y_max, x_max = self.img.shape 						#hessian
		self.extrema = []									#final shape:(4,3,list)
		for i in range(self.octaves):
			scale_factor = 2**(-1+i)
			self.extrema.append([])
			row,col = self.dog[i][0].shape
			for j in range(2,2+self.scales):
				image = self.dog[i][j-2,:,:]				#hessian
			#	plt.figure()
			#	plt.imshow(img,plt.get_cmap('gray'))
				self.extrema[i].append([])
				for x in range(1,row-1):
					for y in range(1,col-1):
						if np.argmax(self.dog[i][j-1:j+2,x-1:x+2,y-1:y+2]) == 13:
							D3_cude = [self.dog[i][j+1][x-1:x+2,y-1:y+2],self.dog[i][j][x-1:x+2,y-1:y+2],self.dog[i][j-1][x-1:x+2,y-1:y+2]]
							keep = derivatives_filter(D3_cude)
							#keep2 = Hessian_filter(image,x_max,y_max,threshold,y,x)
							if keep:# and keep2:
								self.extrema[i][j-2].append((x,y,self.prior*(2**i)*(self.factor**(j))))
			#				plt.plot(y*scale_factor, x*scale_factor,marker='o', markerfacecolor='none',markeredgecolor='r')
			#	plt.show()

	def assign_orientation_all(self):			#self.mag and self.org is from 1 to shape-1
		mag = []
		ori = []
		for i in range(self.octaves):
			mag.append([])
			ori.append([])
			for j in range(2,2+self.scales):
				m,o = self.calculate_MO(i,j)
				mag[i].append(m)
				ori[i].append(o)
		return mag, ori

	def calculate_MO(self,i,j):
		m = []
		o = []
		img = self.octaves_[i][j]
		row,col = img.shape
		for x in range(1,row-1):
			m.append([])
			o.append([])
			for y in range(1,col-1):
				m[x-1].append(((img[x+1,y]-img[x-1,y])**2+(img[x,y+1]-img[x,y-1])**2)**0.5)
				o[x-1].append(handle_o(img[x,y+1]-img[x,y-1],img[x+1,y]-img[x-1,y]))
		return np.array(m),np.array(o)

	def descriptor(self):
		##print('Processing descriptor...')
		shape_ = int((25-1)/2)	#initial 25*25 square, before rotate
		self.descriptor_ = []
		for i in range(self.octaves):
			for j in range(self.scales):
				img = self.octaves_[i][j]
				for ind,(x,y,s,m,o) in enumerate(self.extrema[i][j]):
					if x in range(shape_+1,img.shape[0]-shape_) and y in range(shape_+1,img.shape[1]-shape_):
						###print(self.mag)
						ro_m = rotate(self.mag[i][j][x-1-shape_:x-1+shape_+1,y-1-shape_:y-1+shape_+1],angle= -math.degrees(o),reshape=False)[shape_-8:shape_+8,shape_-8:shape_+8]
						ro_o = rotate(self.ori[i][j][x-1-shape_:x-1+shape_+1,y-1-shape_:y-1+shape_+1],angle= -math.degrees(o),reshape=False)[shape_-8:shape_+8,shape_-8:shape_+8]
						#self.extrema[i][j][ind] += get_descriptor(ro_m,ro_o,o)
						tmp = [int(x*2**(i-1)),int(y*2**(i-1))]+get_descriptor(ro_m,ro_o)
						self.descriptor_.append(tmp)
		self.descriptor_ = np.array(self.descriptor_)

	def Orientation(self):
		##print("=" * 50)
		##print("Orientation & Magnitude Assignment")
		extrema_filtered = []
		for i in range(self.octaves):
			extrema_filtered.append([])
			for j in range(self.scales):
				image = self.dog[i][j, :, :]
				y_max, x_max = image.shape
				sig = self.extrema[i][j][0][2]
				#print(sig)
				mask_size = int(np.ceil(self.orientation_mask_ratio * sig))
				x = np.linspace(-mask_size, mask_size, endpoint=True, num=2 * mask_size + 1)
				y = np.linspace(-mask_size, mask_size, endpoint=True, num=2 * mask_size + 1)
				X, Y = np.meshgrid(x, y)
				d = np.add(np.multiply(X, X), np.multiply(Y, Y))
				mask = np.exp(-(d / (2.0 * sig ** 2))) / (2 * np.pi * sig ** 2)
				extrema_filtered[i].append([])
				for y,x,s in self.extrema[i][j]:
					#y, x = self.extrema[i][j][k][0], self.extrema[i][j][k][1]

					if x < mask_size + 1 or  x > x_max - mask_size - 2:
						continue
					if y < mask_size + 1 or  y > y_max - mask_size - 2:
						continue
					sub_image = image[y - mask_size - 1: y + mask_size + 2, x - mask_size - 1: x + mask_size + 2]

					Dx = 0.5 * (sub_image[1:-1, 2:] - sub_image[1:-1, :-2])
					Dy = 0.5 * (sub_image[2:, 1:-1] - sub_image[:-2, 1:-1])
					magnitude = np.sqrt(np.multiply(Dx, Dx), np.multiply(Dy, Dy))
					orientation = np.arctan(np.divide(Dy, Dx))
					#if orientation.shape[0] != orientation.shape[1]:
						#print("Orientation error, keypoints at boundary")
					for p in range(2 * mask_size + 1):
						for q in range(2 * mask_size + 1):
							if np.isnan(orientation[p, q]):
								orientation[p, q] = 0
								continue
							if np.sign(Dx[p, q]) == -1:
								orientation[p, q] += np.pi
								continue
							if np.sign(Dy[p, q]) == -1:
								orientation[p, q] += np.pi * 2
								continue
					weight = np.multiply(magnitude, mask)
					quantize = np.divide(orientation, 2 * np.pi / self.orientation_bin_number)
					quantize = quantize.astype(int)
					bins = np.zeros(self.orientation_bin_number)
					for p in range(2 * mask_size + 1):
						for q in range(2 * mask_size + 1):
							b = quantize[p, q]
							bins[b] += weight[p, q]
					max_direction = list(bins).index(max(list(bins)))
					max_weight = max(list(bins))
					#self.extrema[i][j][k] += (max_direction,)
					for p in range(len(bins)):
						if bins[p] >= 0.8 * max_weight:
							extrema_filtered[i][j].append((y,x,s,bins[p],max_direction))
							#self.extrema[i][j][k] += (bins[p],)
		#print("keypoints magnitude and orientation assignment finished.")
		#print("=" * 50)
		return extrema_filtered

def derivatives_filter(data): #data will be 3*3*3
	dx = data[1][1,2]-data[1][1,1]
	dy = data[1][2,1]-data[1][1,1]
	dz = data[2][1,1]-data[1][1,1]
	dxdx = data[1][1,2]-2*data[1][1,1]+data[1][1,0]
	dydy = data[1][2,1]-2*data[1][1,1]+data[1][0,1]
	dzdz = data[2][1,1]-2*data[1][1,1]+data[0][1,1]
	dxdy = (data[1][2,2]-data[1][1,2]) - (data[1][2,1]-data[1][1,1])
	dxdz = (data[2][1,2]-data[1][1,2]) - (data[2][1,1]-data[1][1,1])
	dydx = dxdy
	dydz = (data[2][2,1]-data[1][2,1]) - (data[2][1,1]-data[1][1,1])
	dzdx = dxdz
	dzdy = dydz

	dDdXt = np.matrix([dx,dy,dz])
	d2DdX2 = np.matrix([[dxdx,dxdy,dxdz],[dydx,dydy,dydz],[dzdx,dzdy,dzdz]])
	Dxhat = data[1][1,1]-0.5*dDdXt*inv(d2DdX2)*dDdXt.transpose()

	return True if abs(Dxhat)>0.03 else False

def Hessian_filter(image,x_max,y_max,threshold,x,y):
	c1, c2, c3, c4 = (x == 0), (x == x_max - 1), (y == 0), (y == y_max - 1)
	if not c1 and not c2:
		Dxx = image[y, x + 1] + image[y, x - 1] - 2.0 * image[y, x]
	elif c1 and not c2:
		Dxx = 2.0 * image[y, 0] - 5.0 * image[y, 1] + 4.0 * image[y, 2] - image[y, 3]
	elif not c1 and c2:
		Dxx = 2.0 * image[y, -1] - 5.0 * image[y, -2] + 4.0 * image[y, -3] - image[y, -4]
	else:
		Dxx = 0
		#print("invalid Dxx")

	if not c3 and not c4:
		Dyy = image[y + 1, x] + image[y - 1, x] - 2.0 * image[y, x]
	elif c3 and not c4:
		Dyy = 2.0 * image[0, x] - 5.0 * image[1, x] + 4.0 * image[2, x] - image[3, x]
	elif not c3 and c4:
		Dyy = 2.0 * image[-1, x] - 5.0 * image[-2, x] + 4.0 * image[-3, x] - image[-4, x]
	else:
		Dyy = 0
		#print("invalide Dyy")


	if not c1 and not c2 and not c3 and not c4:
		Dxy = image[y + 1, x + 1] + image[y - 1, x - 1] - image[y - 1, x + 1] - image[y + 1, x - 1]
	elif c1 and not c2 and c3 and not c4:
		Dxy = image[2, 2] + 16 * image[1, 1] + 9 * image[0, 0] - 4 * (image[1, 2] + image[2, 1]) \
			  - 12 * (image[1, 0] + image[0, 1]) + 3 * (image[0, 2] + image[2, 0])
	elif not c1 and not c2 and c3 and not c4:
		Dxy = -1.0 * (image[2, x + 1] - image[2, x - 1]) + 4.0 * (image[1, x + 1] - image[1, x-1]) + \
			  -3.0 * (image[0, x + 1] - image[0, x - 1] )
	elif not c1 and c2 and c3 and not c4:
		Dxy = -1 * image[2, -3] - 16 * image[1, -2] - 9 * image[0, -1] + \
			   4 * (image[1, -3] + image[2, -2]) + 12 * (image[1, -1] + image[0, -2]) - \
			   3 * (image[0, -3] + image[2, -1])
	elif c1 and not c2 and not c3 and not c4:
		Dxy = -1.0 * (image[y + 1, 2] - image[y - 1, 2]) + 4.0 * (image[y + 1, 1] - image[y - 1, 1]) + \
			  -3.0 * (image[y + 1, 0] - image[y - 1, 0])
	elif not c1 and c2 and not c3 and not c4:
		Dxy = (image[y + 1, -3] - image[y - 1, -3]) + 3.0 * (image[y + 1, -1] - image[y - 1, -1]) + \
			  -4.0 * (image[y + 1, -2] - image[y - 1, -2])
	elif c1 and not c2 and not c3 and c4:
		Dxy = -1 * image[-3, 2] - 16 * image[-2, 1] - 9 * image[-1, 0] + \
			  4 * (image[-3, 1] + image[-2, 2]) + 12 * (image[-2, 0] + image[-1, 1]) - \
			  3 * (image[-3, 0] + image[-1, 2])
	elif not c1 and not c2 and not c3 and c4:
		Dxy = image[-3, x + 1] - image[-3, x - 1] + 3.0 * (image[-1, x + 1] - image[-1, x - 1] ) + \
			  -4.0 * (image[-2, x + 1] - image[-2, x - 1])
	elif not c1 and c2 and not c3 and c4:
		Dxy = image[-3, -3] + 16 * image[-2, -2] + 9 * image[-1, -1] + \
			  -4 * (image[-2, -3] + image[-3, -2]) - 12 * (image[-1, -2] + image[-2, -2]) + \
			  3 * (image[-1, -3] + image[-3, -1])
	else:
		Dxy = 0
		#print("invalide Dxy and Dyx")
	Dxy *= 0.25
	Dyx = Dxy

	H = np.asarray([[Dxx, Dxy], [Dyx, Dyy]])
	t = np.trace(H)
	d = np.linalg.det(H)
	if d == 0:
		#print("=" * 20)
		##print("warning: octave = ", str(i), ", level = ", str(j), ", x = ", str(x), ", y = ", str(y))
		#print("Dxx = ", str(Dxx), ", Dyy = ", str(Dyy), ", Dxy = Dyx = ", str(Dxy), ", DET = 0 !!")
		return False

	check = abs(t * t / d)
	return True if check <= threshold else False

def get_descriptor(m,o):							#16 by 16 block
	#visualize_orientation(m,o)
	sig = int(16/2)
	gau = kernel(sig)
	cent = 2*sig
	m_ = m*gau[cent-8:cent+8,cent-8:cent+8]			#magnitude weighted by gaussian
	descriptor = []
	for i in range(4):
		#descriptor.append([])
		for j in range(4):
			descriptor += get_4by4_patch(m_[i*4:(i+1)*4,j*4:(j+1)*4],o[i*4:(i+1)*4,j*4:(j+1)*4])
	norm = LA.norm(descriptor, 2)
	descriptor = [i/norm for i in descriptor]
	#visualize_descriptor(descriptor)
	return descriptor

def get_4by4_patch(m,o):						#4 by 4 patch orientation and magnitude
	des_ = [0,0,0,0,0,0,0,0]
	d_bar = np.zeros(o.shape)
	for i in range(4):
		for j in range(4):
			d_bar[i,j] = 1 - abs(o[i,j])%(2*np.pi)/np.pi if abs(o[i,j])%(2*np.pi) <=np.pi else 1 - abs(o[i,j]%(2*np.pi)-np.pi)/np.pi
	weighted_m = m*d_bar
	o_discrete = np.round(o/(np.pi/4)).astype('int')
	for i in range(4):
		for j in range(4):
			des_[o_discrete[i,j]%8] += weighted_m[i,j]
	return des_

def visualize_orientation(m,o):
	#print(m)
	#print(o)
	fig, axes = plt.subplots(16, 16, figsize=(4,4), gridspec_kw = {'wspace':0, 'hspace':0})
	for i, ax in enumerate(fig.axes):
		i_ = i//16
		j_ = i%16
		plt.setp(ax.spines.values(), color='green')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.axis([-1,1,-1,1])
		ax.tick_params( axis='both', which='both', bottom='off', left='off', labelleft='off', labelbottom='off')
		#des_ = des[int(i_)][int(j_)]
		o_ = o[int(i_),int(j_)]
		m_ = m[int(i_),int(j_)]
		ax.arrow(0, 0, m_*math.cos(o_), m_*math.sin(o_), head_width=0.3, fc='k', ec='k')
		ax.set_aspect('equal')
	plt.show()

def visualize_descriptor(des):
	fig, axes = plt.subplots(4, 4, figsize=(4,4), gridspec_kw = {'wspace':0, 'hspace':0})
	for i, ax in enumerate(fig.axes):
		plt.setp(ax.spines.values(), color='green')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.axis([-0.1,0.1,-0.1,0.1])
		ax.tick_params( axis='both', which='both', bottom='off', left='off', labelleft='off', labelbottom='off')
		#des_ = des[int(i_)][int(j_)]
		des_ = des[i*8:(i+1)*8]
		for ind,d in enumerate(des_):
				ax.arrow(0, 0, d*math.cos(ind*np.pi/4), d*math.sin(ind*np.pi/4), head_width=0.01, fc='k', ec='k')
		ax.set_aspect('equal')
	plt.show()

def handle_o(num,den):
	if den > 0:
		return math.atan(num/den) if num >= 0 else math.atan(num/den)+2*np.pi
	elif den == 0:
		return np.pi/2 if num>=0 else np.pi*1.5
	else:
		return math.atan(num/den)+np.pi

def main():
	img = np.array(Image.open('image.jpg'))
	#print ('image_shape=',img.shape)
	sift = SIFT_(img)
	features = sift.get_features()


if __name__ == '__main__':
	main()
