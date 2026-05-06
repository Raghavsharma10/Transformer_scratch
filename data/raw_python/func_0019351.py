def create_variable(ncfile, name, datatype, dimensions) -> None:
    """Add a new variable with the given name, datatype, and dimensions
    to the given NetCDF file.

    Essentially, |create_variable| just calls the equally named method
    of the NetCDF library, but adds information to possible error messages:

    >>> from hydpy import TestIO
    >>> from hydpy.core.netcdftools import netcdf4
    >>> with TestIO():
    ...     ncfile = netcdf4.Dataset('test.nc', 'w')
    >>> from hydpy.core.netcdftools import create_variable
    >>> try:
    ...     create_variable(ncfile, 'var1', 'f8', ('dim1',))
    ... except BaseException as exc:
    ...     print(str(exc).strip('"'))    # doctest: +ELLIPSIS
    While trying to add variable `var1` with datatype `f8` and \
dimensions `('dim1',)` to the NetCDF file `test.nc`, the following error \
occurred: ...

    >>> from hydpy.core.netcdftools import create_dimension
    >>> create_dimension(ncfile, 'dim1', 5)
    >>> create_variable(ncfile, 'var1', 'f8', ('dim1',))
    >>> import numpy
    >>> numpy.array(ncfile['var1'][:])
    array([ nan,  nan,  nan,  nan,  nan])

    >>> ncfile.close()
    """
    default = fillvalue if (datatype == 'f8') else None
    try:
        ncfile.createVariable(
            name, datatype, dimensions=dimensions, fill_value=default)
        ncfile[name].long_name = name
    except BaseException:
        objecttools.augment_excmessage(
            'While trying to add variable `%s` with datatype `%s` '
            'and dimensions `%s` to the NetCDF file `%s`'
            % (name, datatype, dimensions, get_filepath(ncfile)))