def from_array(cls, array):
        """Return a |Date| instance based on date information (year,
        month, day, hour, minute, second) stored as the first entries of
        the successive rows of a |numpy.ndarray|.

        >>> from hydpy import Date
        >>> import numpy
        >>> array1d = numpy.array([1992, 10, 8, 15, 15, 42, 999])
        >>> Date.from_array(array1d)
        Date('1992-10-08 15:15:42')

        >>> array3d = numpy.zeros((7, 2, 2))
        >>> array3d[:, 0, 0] = array1d
        >>> Date.from_array(array3d)
        Date('1992-10-08 15:15:42')

        .. note::

           The date defined by the given |numpy.ndarray| cannot
           include any time zone information and corresponds to
           |Options.utcoffset|, which defaults to UTC+01:00.
        """
        intarray = numpy.array(array, dtype=int)
        for dummy in range(1, array.ndim):
            intarray = intarray[:, 0]
        return cls(datetime.datetime(*intarray[:6]))