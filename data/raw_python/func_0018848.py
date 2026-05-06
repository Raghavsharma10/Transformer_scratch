def verify(self) -> None:
        """Raises a |RuntimeError| if at least one of the required values
        of a |Variable| object is |None| or |numpy.nan|. The descriptor
        `mask` defines, which values are considered to be necessary.

        Example on a 0-dimensional |Variable|:

        >>> from hydpy.core.variabletools import Variable
        >>> class Var(Variable):
        ...     NDIM = 0
        ...     TYPE = float
        ...     __hydpy__connect_variable2subgroup__ = None
        ...     initinfo = 0.0, False
        >>> var = Var(None)
        >>> import numpy
        >>> var.shape = ()
        >>> var.value = 1.0
        >>> var.verify()
        >>> var.value = numpy.nan
        >>> var.verify()
        Traceback (most recent call last):
        ...
        RuntimeError: For variable `var`, 1 required value has not been set yet.

        Example on a 2-dimensional |Variable|:

        >>> Var.NDIM = 2
        >>> var = Var(None)
        >>> var.shape = (2, 3)
        >>> var.value = numpy.ones((2,3))
        >>> var.value[:, 1] = numpy.nan
        >>> var.verify()
        Traceback (most recent call last):
        ...
        RuntimeError: For variable `var`, 2 required values \
have not been set yet.

        >>> Var.mask = var.mask
        >>> Var.mask[0, 1] = False
        >>> var.verify()
        Traceback (most recent call last):
        ...
        RuntimeError: For variable `var`, 1 required value has not been set yet.

        >>> Var.mask[1, 1] = False
        >>> var.verify()
        """
        nmbnan: int = numpy.sum(numpy.isnan(
            numpy.array(self.value)[self.mask]))
        if nmbnan:
            if nmbnan == 1:
                text = 'value has'
            else:
                text = 'values have'
            raise RuntimeError(
                f'For variable {objecttools.devicephrase(self)}, '
                f'{nmbnan} required {text} not been set yet.')