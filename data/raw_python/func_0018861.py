def compress_repr(self) -> Optional[str]:
        """Try to find a compressed parameter value representation and
        return it.

        |Parameter.compress_repr| raises a |NotImplementedError| when
        failing to find a compressed representation.

        .. testsetup::

            >>> from hydpy import pub
            >>> del pub.timegrids

        For the following examples, we define a 1-dimensional sequence
        handling time-dependent floating point values:

        >>> from hydpy.core.parametertools import Parameter
        >>> class Test(Parameter):
        ...     NDIM = 1
        ...     TYPE = float
        ...     TIME = True
        >>> test = Test(None)

        Before and directly after defining the parameter shape, `nan`
        is returned:

        >>> test.compress_repr()
        '?'
        >>> test
        test(?)
        >>> test.shape = 4
        >>> test
        test(?)

        Due to the time-dependence of the values of our test class,
        we need to specify a parameter and a simulation time step:

        >>> test.parameterstep = '1d'
        >>> test.simulationstep = '8h'

        Compression succeeds when all required values are identical:

        >>> test(3.0, 3.0, 3.0, 3.0)
        >>> test.values
        array([ 1.,  1.,  1.,  1.])
        >>> test.compress_repr()
        '3.0'
        >>> test
        test(3.0)

        Method |Parameter.compress_repr| returns |None| in case the
        required values are not identical:

        >>> test(1.0, 2.0, 3.0, 3.0)
        >>> test.compress_repr()
        >>> test
        test(1.0, 2.0, 3.0, 3.0)

        If some values are not required, indicate this by the `mask`
        descriptor:

        >>> import numpy
        >>> test(3.0, 3.0, 3.0, numpy.nan)
        >>> test
        test(3.0, 3.0, 3.0, nan)
        >>> Test.mask = numpy.array([True, True, True, False])
        >>> test
        test(3.0)

        For a shape of zero, the string representing includes an empty list:

        >>> test.shape = 0
        >>> test.compress_repr()
        '[]'
        >>> test
        test([])

        Method |Parameter.compress_repr| works similarly for different
        |Parameter| subclasses.  The following examples focus on a
        2-dimensional parameter handling integer values:

        >>> from hydpy.core.parametertools import Parameter
        >>> class Test(Parameter):
        ...     NDIM = 2
        ...     TYPE = int
        ...     TIME = None
        >>> test = Test(None)

        >>> test.compress_repr()
        '?'
        >>> test
        test(?)
        >>> test.shape = (2, 3)
        >>> test
        test(?)

        >>> test([[3, 3, 3],
        ...       [3, 3, 3]])
        >>> test
        test(3)

        >>> test([[3, 3, -999999],
        ...       [3, 3, 3]])
        >>> test
        test([[3, 3, -999999],
              [3, 3, 3]])

        >>> Test.mask = numpy.array([
        ...     [True, True, False],
        ...     [True, True, True]])
        >>> test
        test(3)

        >>> test.shape = (0, 0)
        >>> test
        test([[]])
        """
        if not hasattr(self, 'value'):
            return '?'
        if not self:
            return f"{self.NDIM * '['}{self.NDIM * ']'}"
        unique = numpy.unique(self[self.mask])
        if sum(numpy.isnan(unique)) == len(unique.flatten()):
            unique = numpy.array([numpy.nan])
        else:
            unique = self.revert_timefactor(unique)
        if len(unique) == 1:
            return objecttools.repr_(unique[0])
        return None