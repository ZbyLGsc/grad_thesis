import numpy as np
import cv2 as cv


# --------------------------main-------------------------------------------

#----------------------------read and show image------------------------
cv.namedWindow('origin', cv.WINDOW_AUTOSIZE)
cv.namedWindow('defect', cv.WINDOW_AUTOSIZE)
cv.moveWindow('origin', 100, 100)
cv.moveWindow('defect', 500, 100)

img = cv.imread('./origin_images/1.png')
# res = cv.resize(img, (300, 600), interpolation=cv.INTER_CUBIC)

cv.imshow('origin', img)

#---------------------------add some defects----------------------------
# we add defect as some points, whose r,g,b=(64,28,11)

# get the cols and rows of image
rows, cols, chs = img.shape
print('rows:', rows, 'cols:', cols)


# generate the defect center and range randomly
center_x = int(np.random.normal(rows / 2, 50.0))
center_y = int(np.random.normal(cols / 2, 50.0))
df_range = int(np.random.normal(4, 0.01))
print('x:', center_x, 'y:', center_y, 'range:', df_range)

# draw defect around the defect center, and record the rectangle around the defect
draw = img
array_x = []
array_y = []

for i in range(1, df_range**2):
    df_x = int(np.random.normal(center_x, df_range))
    df_y = int(np.random.normal(center_y, df_range))
    # cv.circle(draw, (df_x, df_y), 2, (11, 28, 64),2)
    cv.circle(draw, (df_x, df_y), 3, (120, 160, 179), 6)

    array_x.append(df_x)
    array_y.append(df_y)

max_x = np.amax(array_x)
min_x = np.amin(array_x)
max_y = np.amax(array_y)
min_y = np.amin(array_y)

# smooth the image around the defect, using filtering
offset = 6
roi_img = draw[(min_y - offset):(max_y +
                                 offset), (min_x - offset):(max_x + offset)]
roi_img = cv.blur(roi_img, (4, 4))

draw[(min_y - offset):(max_y +
                       offset), (min_x - offset):(max_x + offset)] = roi_img

# show result
cv.imshow('defect', draw)

# --------------------save defect image and set labels--------------------------
name = "./generated_images/spotted.png"
cv.imwrite(name, draw)


cv.waitKey(0)
