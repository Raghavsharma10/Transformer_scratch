def class_error(self, input_data, targets, average=True,
                    cache=None, prediction=False):
        """ Return the classification error rate
        """

        if cache is not None:
            activations = cache
        else:
            activations = \
              self.feed_forward(input_data, prediction=prediction)

        targets = targets.get().argmax(1)
        class_error = np.sum(activations.get().argmax(1) != targets)

        if average: class_error = float(class_error) / targets.shape[0]
        return class_error