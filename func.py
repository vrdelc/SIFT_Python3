import numpy as np
from PIL import Image
from pathlib import Path
import math
from scipy.signal import convolve2d

def get_image(file_name):
	file = Path('img_data.npy')
	if file.exists():
		data = np.load('img_data.npy')
		print ('File exists.')
	else:
		im = open(file_name)
		lines = im.readlines()
		width,height = [int(i) for i in lines[2].split(' ')]
		raw_data = lines[4:]
		data = np.array([int(i.strip('\n')) for i in raw_data]).reshape((height,width))
		np.save('img_data.npy',data)
		print (data)
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
	print ('Gaussian')
	g = kernel(sigma)
	return overflow(convolve2d(im, g,mode='same',boundary='symm')/sum(sum(g)))


class SIFT:
	def __init__(self,img):
		self.origin_img = img
		self.img = np.array(Image.fromarray(img.astype('uint8'),'L').resize(np.multiply(2,[img.shape[1],img.shape[0]]),resample=Image.BICUBIC))
		self.octaves = 4
		self.scales = 3
		self.factor = 2**0.2
		self.prior = 1.6

		self.DoG()
		self.extrema()
	def DoG(self):
		octaves_ = []
		self.levels = self.scales+3
		for octave in range(self.octaves):
			octaves_.append([Gaussian_filter(self.img, self.prior*(self.factor**level)) for level in range(self.levels)])
			self.img = self.img[::2,::2]
		self.dog = [np.array([ np.subtract(imgs[ind+1],img) for ind,img in enumerate(imgs) if ind != len(imgs)-1])for imgs in octaves_]
		print ('dog[0].shape:',self.dog[0].shape)
		#for i in self.dog:
		#	for j in i:
		#		Image.fromarray(j.astype('uint8'), 'L').show()
	def extrema(self):
		self.extrema = []
		for octave in range(self.octaves):
			self.extrema.append([])
			row,col = self.dog[octave][0].shape
			for level in range(1,1+self.scales):
				self.extrema[octave].append([])
				for x in range(1,row-1):
					for y in range(1,col-1):
						if np.argmax(self.dog[octave][level-1:level+2,x-1:x+2,y-1:y+2]) == 13:
							self.extrema[octave][level-1].append((x,y,self.prior*(self.factor**(level))))
		print (self.extrema)

def main():
	img = np.array(Image.open('example.png'))
	print ('image_shape=',img.shape)
	sift = SIFT(img)








if __name__ == '__main__':
	main()