def average_series(self, *args, **kwargs) -> InfoArray:
        """Average the actual time series of the |Variable| object for all
        time points.

        Method |IOSequence.average_series| works similarly as method
        |Variable.average_values| of class |Variable|, from which we
        borrow some examples. However, firstly, we have to prepare a
        |Timegrids| object to define the |IOSequence.series| length:

        >>> from hydpy import pub
        >>> pub.timegrids = '2000-01-01', '2000-01-04', '1d'

        As shown for method |Variable.average_values|, for 0-dimensional
        |IOSequence| objects the result of |IOSequence.average_series|
        equals |IOSequence.series| itself:

        >>> from hydpy.core.sequencetools import IOSequence
        >>> class SoilMoisture(IOSequence):
        ...     NDIM = 0
        >>> sm = SoilMoisture(None)
        >>> sm.activate_ram()
        >>> import numpy
        >>> sm.series = numpy.array([190.0, 200.0, 210.0])
        >>> sm.average_series()
        InfoArray([ 190.,  200.,  210.])

        For |IOSequence| objects with an increased dimensionality, a
        weighting parameter is required, again:

        >>> SoilMoisture.NDIM = 1
        >>> sm.shape = 3
        >>> sm.activate_ram()
        >>> sm.series = (
        ...     [190.0, 390.0, 490.0],
        ...     [200.0, 400.0, 500.0],
        ...     [210.0, 410.0, 510.0])
        >>> from hydpy.core.parametertools import Parameter
        >>> class Area(Parameter):
        ...     NDIM = 1
        ...     shape = (3,)
        ...     value = numpy.array([1.0, 1.0, 2.0])
        >>> area = Area(None)
        >>> SoilMoisture.refweights = property(lambda self: area)
        >>> sm.average_series()
        InfoArray([ 390.,  400.,  410.])

        The documentation on method |Variable.average_values| provides
        many examples on how to use different masks in different ways.
        Here we restrict ourselves to the first example, where a new
        mask enforces that |IOSequence.average_series| takes only the
        first two columns of the `series` into account:

        >>> from hydpy.core.masktools import DefaultMask
        >>> class Soil(DefaultMask):
        ...     @classmethod
        ...     def new(cls, variable, **kwargs):
        ...         return cls.array2mask([True, True, False])
        >>> SoilMoisture.mask = Soil()
        >>> sm.average_series()
        InfoArray([ 290.,  300.,  310.])
        """
        try:
            if not self.NDIM:
                array = self.series
            else:
                mask = self.get_submask(*args, **kwargs)
                if numpy.any(mask):
                    weights = self.refweights[mask]
                    weights /= numpy.sum(weights)
                    series = self.series[:, mask]
                    axes = tuple(range(1, self.NDIM+1))
                    array = numpy.sum(weights*series, axis=axes)
                else:
                    return numpy.nan
            return InfoArray(array, info={'type': 'mean'})
        except BaseException:
            objecttools.augment_excmessage(
                'While trying to calculate the mean value of '
                'the internal time series of sequence %s'
                % objecttools.devicephrase(self))