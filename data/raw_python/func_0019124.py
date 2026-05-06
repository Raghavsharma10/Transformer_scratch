def plot(self, threshold=None, **kwargs):
        """Plot the instanteneous unit hydrograph.

        The optional argument allows for defining a threshold of the cumulative
        sum uf the hydrograph, used to adjust the largest value of the x-axis.
        It must be a value between zero and one.
        """
        delays, responses = self.delay_response_series
        pyplot.plot(delays, responses, **kwargs)
        pyplot.xlabel('time')
        pyplot.ylabel('response')
        if threshold is not None:
            threshold = numpy.clip(threshold, 0., 1.)
            cumsum = numpy.cumsum(responses)
            idx = numpy.where(cumsum >= threshold*cumsum[-1])[0][0]
            pyplot.xlim(0., delays[idx])