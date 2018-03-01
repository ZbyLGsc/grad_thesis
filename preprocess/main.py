import numpy as np
import cv2 as cv
import add_defect as ad

for n in range(1,100):
	m = int(np.random.uniform(1,3369))
	print(n,m)
	
	df = ad.AddDefect(m,n)
	#df.add_dot()
	#df.add_spot()
	df.add_cut()