def rollsd(self, scale=1, **kwargs):
        '''A :ref:`rolling function <rolling-function>` for
        stadard-deviation values:
        Same as::

            self.rollapply('sd', **kwargs)
        '''
        ts = self.rollapply('sd', **kwargs)
        if scale != 1:
            ts *= scale
        return ts