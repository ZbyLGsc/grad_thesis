import model_interface as mi
import os
import time


def eval():
    # data to be calculate
    image_num = 0
    correct_num = 0
    false_positive = 0
    false_negative = 0
    wrong_type = 0
    total_time = 0.0

    # result matrix, y is truth and x is prediction
    res_matrix = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]

    # initialize model
    model_inception = mi.ModelInterface('Inception')
    model_mobilenet = mi.ModelInterface('Mobilenet')

    # specify root dir of the validation images
    root_dir = '/home/zby/Downloads/datasets/board_defect/test'

    # find sub dir names under root dir
    class_names = os.listdir(root_dir)

    #  traverse all image in subdir
    for class_name in class_names:
        class_path = os.path.join(root_dir, class_name)
        image_names = os.listdir(class_path)

        for image_name in image_names:
            image_path = os.path.join(class_path, image_name)
            image_num += 1

            if abs(image_num % 1) < 1e-3:
                print 'Finished num:', image_num

            # use model to label image, while recording time
            time1 = time.time()

            # inception prediction
            label, prob = model_inception.predict(image_path)
            print '\n--------------------'
            print 'Inception:'
            print label
            print prob
            print

            # mobilenet prediction
            label2, prob2 = model_mobilenet.predict(image_path)
            print 'Mobilenet:'
            print label2
            print prob2
            print '--------------------\n'

            time2 = time.time()
            total_time += (time2 - time1)
            # print (time2-time1)

            # calculate some data
            if label[0] == class_name:
                correct_num += 1
            # False positive: truth is normal but result shows defect
            elif class_name == 'normal':
                false_positive += 1
            # False negative: truth has defect but result shows normal
            elif label[0] == 'normal':
                false_negative += 1
            # Wrong type of defect
            else:
                wrong_type += 1

            res_matrix[class_names.index(class_name)][class_names.index(label[0])] += 1

    # print final results
    print 'image_num:', image_num
    print 'correct:', correct_num
    print 'false positive:', false_positive
    print 'false nagetive:', false_negative
    print 'wrong type:', wrong_type

    print 'Res Matrix:'
    print class_names
    for row in res_matrix:
        print row

    # average prediction time
    avg_time = total_time/image_num
    print 'average time:', avg_time


def main():
    print "main:"
    eval()
    # test()


if __name__ == '__main__':
    main()
