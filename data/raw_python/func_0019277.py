def to_array(self):
        """Returns a 1-dimensional |numpy| |numpy.ndarray| with thirteen
        entries first defining the start date, secondly defining the end
        date and thirdly the step size in seconds.
        """
        values = numpy.empty(13, dtype=float)
        values[:6] = self.firstdate.to_array()
        values[6:12] = self.lastdate.to_array()
        values[12] = self.stepsize.seconds
        return values