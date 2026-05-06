def query_timegrid(ncfile) -> timetools.Timegrid:
    """Return the |Timegrid| defined by the given NetCDF file.

    >>> from hydpy.core.examples import prepare_full_example_1
    >>> prepare_full_example_1()
    >>> from hydpy import TestIO
    >>> from hydpy.core.netcdftools import netcdf4
    >>> from hydpy.core.netcdftools import query_timegrid
    >>> filepath = 'LahnH/series/input/hland_v1_input_t.nc'
    >>> with TestIO():
    ...     with netcdf4.Dataset(filepath) as ncfile:
    ...         query_timegrid(ncfile)
    Timegrid('1996-01-01 00:00:00',
             '2007-01-01 00:00:00',
             '1d')
    """
    timepoints = ncfile[varmapping['timepoints']]
    refdate = timetools.Date.from_cfunits(timepoints.units)
    return timetools.Timegrid.from_timepoints(
        timepoints=timepoints[:],
        refdate=refdate,
        unit=timepoints.units.strip().split()[0])