def classify(self, phrase_vector):
        """ Run this over an input vector and see the result """
        x = Variable(np.asarray([phrase_vector]))
        return self.model.predictor(x).data[0]