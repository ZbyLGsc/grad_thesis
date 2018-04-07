import ensemble_interface as ei
import tensorflow as tf
import os


def main():

    dir = '/home/zby/Downloads/datasets/board_defect/true'
    names = os.listdir(dir)

    model = ei.EnsembleModel()
    graph1 = model.model_inception.modelGraph()
    graph2 = model.model_mobilenet.modelGraph()

    mat = [[0, 0], [0, 0]]
    num = 0

    with tf.Session(graph=graph1) as sess1:
        with tf.Session(graph=graph2) as sess2:

            for name in names:
                num += 1
                print num    

                filename = os.path.join(dir, name)

                defect = model.predict(filename, sess1, sess2)

                if defect:
                    mat[1][1] += 1
                else:
                    mat[1][0] += 1
                    print 'name:', filename

            print 'mat', mat


if __name__ == '__main__':
    main()
