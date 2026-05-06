def cross_entropy_error(self, input_data, targets, average=True,
                            cache=None, prediction=False):
        """ Return the cross entropy error
        """

        if cache is not None:
            activations = cache
        else:
            activations = \
              self.feed_forward(input_data, prediction=prediction)

        loss = cross_entropy_logistic(activations, targets)

        if average: loss /= targets.shape[0]
        # assert np.isfinite(loss)
        return loss.get()