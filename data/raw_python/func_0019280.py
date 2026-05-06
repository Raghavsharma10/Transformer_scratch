def array2series(self, array):
        """Prefix the information of the actual Timegrid object to the given
        array and return it.

        The Timegrid information is stored in the first thirteen values of
        the first axis of the returned series.  Initialize a Timegrid object
        and apply its `array2series` method on a simple list containing
        numbers:

        >>> from hydpy import Timegrid
        >>> timegrid = Timegrid('2000-11-01 00:00', '2000-11-01 04:00', '1h')
        >>> series = timegrid.array2series([1, 2, 3.5, '5.0'])

        The first six entries contain the first date of the timegrid (year,
        month, day, hour, minute, second):

        >>> from hydpy import round_
        >>> round_(series[:6])
        2000.0, 11.0, 1.0, 0.0, 0.0, 0.0

        The six subsequent entries contain the last date:

        >>> round_(series[6:12])
        2000.0, 11.0, 1.0, 4.0, 0.0, 0.0

        The thirteens value is the step size in seconds:

        >>> round_(series[12])
        3600.0

        The last four value are the ones of the given vector:

        >>> round_(series[-4:])
        1.0, 2.0, 3.5, 5.0

        The given array can have an arbitrary number of dimensions:

        >>> import numpy
        >>> array = numpy.eye(4)
        >>> series = timegrid.array2series(array)

        Now the timegrid information is stored in the first column:

        >>> round_(series[:13, 0])
        2000.0, 11.0, 1.0, 0.0, 0.0, 0.0, 2000.0, 11.0, 1.0, 4.0, 0.0, 0.0, \
3600.0

        All other columns of the first thirteen rows contain nan values, e.g.:

        >>> round_(series[12, :])
        3600.0, nan, nan, nan

        The original values are stored in the last four rows, e.g.:

        >>> round_(series[13, :])
        1.0, 0.0, 0.0, 0.0

        Inappropriate array objects result in error messages like:

        >>> timegrid.array2series([[1, 2], [3]])
        Traceback (most recent call last):
        ...
        ValueError: While trying to prefix timegrid information to the given \
array, the following error occurred: setting an array element with a sequence.

        If the given array does not fit to the defined timegrid, a special
        error message is returned:

        >>> timegrid.array2series([[1, 2], [3, 4]])
        Traceback (most recent call last):
        ...
        ValueError: When converting an array to a sequence, the lengths of \
the timegrid and the given array must be equal, but the length of the \
timegrid object is `4` and the length of the array object is `2`.
        """
        try:
            array = numpy.array(array, dtype=float)
        except BaseException:
            objecttools.augment_excmessage(
                'While trying to prefix timegrid information to the '
                'given array')
        if len(array) != len(self):
            raise ValueError(
                f'When converting an array to a sequence, the lengths of the '
                f'timegrid and the given array must be equal, but the length '
                f'of the timegrid object is `{len(self)}` and the length of '
                f'the array object is `{len(array)}`.')
        shape = list(array.shape)
        shape[0] += 13
        series = numpy.full(shape, numpy.nan)
        slices = [slice(0, 13)]
        subshape = [13]
        for dummy in range(1, series.ndim):
            slices.append(slice(0, 1))
            subshape.append(1)
        series[tuple(slices)] = self.to_array().reshape(subshape)
        series[13:] = array
        return series