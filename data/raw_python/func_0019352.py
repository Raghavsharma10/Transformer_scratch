def query_variable(ncfile, name) -> netcdf4.Variable:
    """Return the variable with the given name from the given NetCDF file.

    Essentially, |query_variable| just performs a key assess via the
    used NetCDF library, but adds information to possible error messages:

    >>> from hydpy.core.netcdftools import query_variable
    >>> from hydpy import TestIO
    >>> from hydpy.core.netcdftools import netcdf4
    >>> with TestIO():
    ...     file_ = netcdf4.Dataset('model.nc', 'w')
    >>> query_variable(file_, 'flux_prec')
    Traceback (most recent call last):
    ...
    OSError: NetCDF file `model.nc` does not contain variable `flux_prec`.

    >>> from hydpy.core.netcdftools import create_variable
    >>> create_variable(file_, 'flux_prec', 'f8', ())
    >>> isinstance(query_variable(file_, 'flux_prec'), netcdf4.Variable)
    True

    >>> file_.close()
    """
    try:
        return ncfile[name]
    except (IndexError, KeyError):
        raise OSError(
            'NetCDF file `%s` does not contain variable `%s`.'
            % (get_filepath(ncfile), name))