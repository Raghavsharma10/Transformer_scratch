def plot(self, threshold=None, **kwargs):
        """Barplot of the ARMA response."""
        try:
            # Works under matplotlib 3.
            pyplot.bar(x=self.ma.delays+.5, height=self.response,
                       width=1., fill=False, **kwargs)
        except TypeError:   # pragma: no cover
            # Works under matplotlib 2.
            pyplot.bar(left=self.ma.delays+.5, height=self.response,
                       width=1., fill=False, **kwargs)
        pyplot.xlabel('time')
        pyplot.ylabel('response')
        if threshold is not None:
            cumsum = numpy.cumsum(self.response)
            idx = numpy.where(cumsum > threshold*cumsum[-1])[0][0]
            pyplot.xlim(0., idx)