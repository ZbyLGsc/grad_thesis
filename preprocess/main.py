import numpy as np
import cv2 as cv
import os
import add_defect as ad

for n in range(1,10):
	m = int(np.random.uniform(1,3369))
	print(n,m)
	
	df = ad.AddDefect(m,n)
	df.add_abrasion()
	df.write_image("./generated_images/abrasion/abrasion_"+str(n)+".png")
"""
	df.add_spot()
	df.add_cut()
	
	df.write_image("./generated_images/defect/df_"+str(n)+".png")
"""