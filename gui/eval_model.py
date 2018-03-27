import model_interface as mi
import os
import time
import tensorflow as tf

# usage: prei = [[labeli], [probi]]


def ensemble_predict(pre1, pre2, pre3=None):
    # probability larger than 50% is judged as unnormal
    threshold = 0.60

    print('pre1: ', pre1)
    print('pre2: ', pre2)

    if pre3 == None:
        if pre1[1][pre1[0].index('normal')] > threshold and pre2[1][pre2[0].index('normal')] > threshold:
            return False
        else:
            return True
    else:
        if pre1[1][pre1[0].index('normal')] > threshold and pre2[1][pre2[0].index('normal')] > threshold and pre3[1][pre3[0].index('normal')] > threshold:
            return False
        else:
            return True


def eval():
    # data to be calculate
    image_num = 0
    correct_num = 0
    false_positive = 0
    false_negative = 0
    wrong_type = 0
    total_time = 0.0
    # [ nor nor, nor defect
    #   def nor, def def
    # ]
    ensem_num = [[0, 0], [0, 0]]

    # result matrix, y is truth and x is prediction
    res_matrix = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]

    # initialize model
    model_inception = mi.ModelInterface('Inception')
    model_mobilenet = mi.ModelInterface('Mobilenet')

    # specify root dir of the validation images
    root_dir = '/home/zby/Downloads/datasets/board_defect/test'
    # root_dir = '/home/sjtu/Downloads/boyu/bancai_images/test'

    # find sub dir names under root dir
    class_names = os.listdir(root_dir)
    print class_names

    graph1 = model_inception.modelGraph()
    with tf.Session(graph=graph1) as sess:
        #  traverse all image in subdir
        for class_name in class_names:
            class_path = os.path.join(root_dir, class_name)
            image_names = os.listdir(class_path)

            for image_name in image_names:
                image_path = os.path.join(class_path, image_name)
                image_num += 1

                print '\n--------------------'
                if abs(image_num % 1) < 1e-3:
                    print 'Finished num:', image_num

                # use model to label image, while recording time
                time1 = time.time()

                # inception prediction
                label, prob = model_inception.predict(image_path, sess)
                print 'Inception:'
                print label
                print prob
                print

                # mobilenet prediction
                # label2, prob2 = model_mobilenet.predict(image_path)
                label2 = label
                prob2 = prob
                print 'Mobilenet:'
                print label2
                print prob2

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

                # ensemble different model
                defect = ensemble_predict([label, prob], [label2, prob2])
                if class_name == 'normal':
                    if not defect:
                        ensem_num[0][0] += 1
                    else:
                        ensem_num[0][1] += 1
                else:
                    if not defect:
                        ensem_num[1][0] += 1
                    else:
                        ensem_num[1][1] += 1
                print 'ensemble:'
                print ensem_num[0]
                print ensem_num[1]
                print '--------------------\n'

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


if __name__ == '__main__':
    main()
