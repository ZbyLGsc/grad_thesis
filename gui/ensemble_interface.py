import model_interface as mi
import tensorflow as tf


class EnsembleModel:

    def __init__(self):

        self.model_inception = mi.ModelInterface('Inception')
        self.model_mobilenet = mi.ModelInterface('Mobilenet')

    def predict(self, filename, sess1, sess2):

        label1, prob1 = self.model_inception.predict(filename, sess1)
        label2, prob2 = self.model_mobilenet.predict(filename, sess2)

        return self.ensemble_predict([label1, prob1], [label2, prob2])


    def ensemble_predict(self, pre1, pre2, pre3=None):
        # probability larger than 50% is judged as unnormal
        threshold = 0.70

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
