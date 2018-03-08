from PIL import Image
from SIFT import SIFT_
import numpy as np













def main():
	img1 = Image.open('images/01.jpg').convert('L')
	img2 = Image.open('images/02.jpg').convert('L')
	
	print(np.array(img1).shape)

	s = SIFT_(np.array(img1))
	f = s.get_features()

	img1.show()
	img2.show()

if __name__ == '__main__':
	main()