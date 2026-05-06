def _product(shape) -> Iterator[Tuple[int, ...]]:
        """Should return all "subdevice index combinations" for sequences
        with arbitrary dimensions:

        >>> from hydpy.core.netcdftools import NetCDFVariableFlat
        >>> _product = NetCDFVariableFlat.__dict__['_product'].__func__
        >>> for comb in _product([1, 2, 3]):
        ...     print(comb)
        (0, 0, 0)
        (0, 0, 1)
        (0, 0, 2)
        (0, 1, 0)
        (0, 1, 1)
        (0, 1, 2)
        """
        return itertools.product(*(range(nmb) for nmb in shape))