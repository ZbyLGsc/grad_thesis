# This script is used to convert .bmp format image to .png format,
# since .bmp is currently not supported in tensorflow
#
#
#

import os
import cv2


def main():
    # dir = '/home/zby/Downloads/datasets/board_defect/image_data/tea/xiao_guang_cha/train_data/PositiveData'
    dir = '/home/zby/Downloads/datasets/board_defect/image_data/bancai'
    new_dir = '/home/zby/Downloads/datasets/board_defect/image_data/png_bancai'

    # get image names list
    names = os.listdir(dir)

    # traverse the dir
    count = 0
    for name in names:
        count += 1
        img_name = os.path.join(dir, name)
        new_name = os.path.join(new_dir, str(count)+'.png')

        img = cv2.imread(img_name)
        # cv2.imshow("win", img)
        # cv2.waitKey(1)
        cv2.imwrite(new_name, img)

        print count


if __name__ == '__main__':
    main()
