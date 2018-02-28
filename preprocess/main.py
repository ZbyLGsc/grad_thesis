import numpy as np
import cv2 as cv
import add_defect as ad

for n in range(1,20):
	m = int(np.random.uniform(1,3369))
	print(m)
	
	dot = ad.AddDefect(m,n)
	dot.add_dot()