def kl_error(self, input_data, targets, average=True,
                 cache=None, prediction=True):
        """ The KL divergence error
        """

        if cache is not None:
            activations = cache
        else:
            activations = \
              self.feed_forward(input_data, prediction=prediction)

        targets_non_nan = gpuarray.empty_like(targets)
        nan_to_zeros(targets, targets_non_nan)
        kl_error = gpuarray.sum(targets_non_nan *
                                (cumath.log(targets_non_nan + eps) -
                                 cumath.log(activations + eps)))
        if average:
            kl_error /= targets.shape[0]
        return kl_error.get()