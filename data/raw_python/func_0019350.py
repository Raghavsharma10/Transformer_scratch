def create_dimension(ncfile, name, length) -> None:
    """Add a new dimension with the given name and length to the given
    NetCDF file.

    Essentially, |create_dimension| just calls the equally named method
    of the NetCDF library, but adds information to possible error messages:

    >>> from hydpy import TestIO
    >>> from hydpy.core.netcdftools import netcdf4
    >>> with TestIO():
    ...     ncfile = netcdf4.Dataset('test.nc', 'w')
    >>> from hydpy.core.netcdftools import create_dimension
    >>> create_dimension(ncfile, 'dim1', 5)
    >>> dim = ncfile.dimensions['dim1']
    >>> dim.size if hasattr(dim, 'size') else dim
    5

    >>> try:
    ...     create_dimension(ncfile, 'dim1', 5)
    ... except BaseException as exc:
    ...     print(exc)    # doctest: +ELLIPSIS
    While trying to add dimension `dim1` with length `5` \
to the NetCDF file `test.nc`, the following error occurred: ...

    >>> ncfile.close()
    """
    try:
        ncfile.createDimension(name, length)
    except BaseException:
        objecttools.augment_excmessage(
            'While trying to add dimension `%s` with length `%d` '
            'to the NetCDF file `%s`'
            % (name, length, get_filepath(ncfile)))