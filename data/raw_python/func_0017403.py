def rollapply(self, func, window=20, **kwargs):
        '''A generic :ref:`rolling function <rolling-function>`
        for function *func*.
        Same construct as :meth:`dynts.TimeSeries.apply` but with default
        ``window`` set to ``20``.
        '''
        return self.apply(func, window=window, **kwargs)