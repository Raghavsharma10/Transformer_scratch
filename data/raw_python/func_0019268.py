def to_array(self):
        """Return a 1-dimensional |numpy| |numpy.ndarray|  with six entries
        defining the actual date (year, month, day, hour, minute, second).

        >>> from hydpy import Date
        >>> Date('1992-10-8 15:15:42').to_array()
        array([ 1992.,    10.,     8.,    15.,    15.,    42.])

        .. note::

           The date defined by the returned |numpy.ndarray| does not
           include any time zone information and corresponds to
           |Options.utcoffset|, which defaults to UTC+01:00.
        """
        return numpy.array([self.year, self.month, self.day, self.hour,
                            self.minute, self.second], dtype=float)