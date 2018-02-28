import numpy as np
import cv2 as cv

r=300 # edges of raw background pics are blank, cut them
c=300
n_pic=1   # ordinal of separate pics
for ii in range(0,10):
	print(ii)
	img = cv.imread('./raw_background/'+str(ii)+'.tif')

	# ------------read image------------
	#img = cv.imread('./raw_background/029.tif')
	row,col,chl = img.shape
	print(img.shape)

	# -------get ranges of small background pictures-----
	_r = list(range(r,row-r,r))
	print(_r)
	_c = list(range(c,col-c,c))
	print(_c)

	# ------output------
	for i in range(0,len(_r)-1):
		for j in range(0,len(_c)-1):
			for k in range(0,10):
				rd = int(np.random.uniform(-100,100))
				pic = img[_r[i]+rd:(_r[i+1])+rd
							, _c[j]+rd:(_c[j+1])+rd]
				name = "./origin_images/"+str(n_pic)+".png"
				cv.imwrite(name, pic)
				n_pic+=1
				cv.destroyAllWindows()
	print(n_pic)
			
cv.destroyAllWindows()