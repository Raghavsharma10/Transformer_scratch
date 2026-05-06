def query_array(ncfile, name) -> numpy.ndarray:
    """Return the data of the variable with the given name from the given
    NetCDF file.

    The following example shows that |query_array| returns |nan| entries
    to represent missing values even when the respective NetCDF variable
    defines a different fill value:

    >>> from hydpy import TestIO
    >>> from hydpy.core.netcdftools import netcdf4
    >>> from hydpy.core import netcdftools
    >>> netcdftools.fillvalue = -999.0
    >>> with TestIO():
    ...     with netcdf4.Dataset('test.nc', 'w') as ncfile:
    ...         netcdftools.create_dimension(ncfile, 'dim1', 5)
    ...         netcdftools.create_variable(ncfile, 'var1', 'f8', ('dim1',))
    ...     ncfile = netcdf4.Dataset('test.nc', 'r')
    >>> netcdftools.query_variable(ncfile, 'var1')[:].data
    array([-999., -999., -999., -999., -999.])
    >>> netcdftools.query_array(ncfile, 'var1')
    array([ nan,  nan,  nan,  nan,  nan])
    >>> import numpy
    >>> netcdftools.fillvalue = numpy.nan
    """
    variable = query_variable(ncfile, name)
    maskedarray = variable[:]
    fillvalue_ = getattr(variable, '_FillValue', numpy.nan)
    if not numpy.isnan(fillvalue_):
        maskedarray[maskedarray.mask] = numpy.nan
    return maskedarray.data