def _differentiate(self, params=None):
        '''Return a sequence of gradients for our parameters.

        If this optimizer has been configured with a gradient norm limit, or
        with elementwise gradient clipping, this method applies the appropriate
        rescaling and clipping operations before returning the gradient.

        Parameters
        ----------
        params : list of Theano variables, optional
            Return the gradient with respect to these parameters. Defaults to
            all parameters that the optimizer knows about.

        Yields
        ------
        pairs : (param, grad) tuples
            Generates a sequence of tuples representing each of the parameters
            requested and the corresponding Theano gradient expressions.
        '''
        if params is None:
            params = self._params
        for param, grad in zip(params, TT.grad(self._loss, params)):
            if self.max_gradient_elem > 0:
                limit = util.as_float(self.max_gradient_elem)
                yield param, TT.clip(grad, -limit, limit)
            elif self.max_gradient_norm > 0:
                norm = TT.sqrt((grad * grad).sum())
                limit = util.as_float(self.max_gradient_norm)
                yield param, grad * TT.minimum(1, limit / norm)
            else:
                yield param, grad