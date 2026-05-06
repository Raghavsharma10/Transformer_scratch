def query_subdevice2index(self, ncfile) -> Subdevice2Index:
        """Return a |Subdevice2Index| that maps the (sub)device names to
        their position within the given NetCDF file.

        Method |NetCDFVariableBase.query_subdevice2index| is based on
        |NetCDFVariableBase.query_subdevices|.  The returned
        |Subdevice2Index| object remembers the NetCDF file the
        (sub)device names stem from, allowing for clear error messages:

        >>> from hydpy.core.netcdftools import NetCDFVariableBase, str2chars
        >>> from hydpy import make_abc_testable, TestIO
        >>> from hydpy.core.netcdftools import netcdf4
        >>> with TestIO():
        ...     ncfile = netcdf4.Dataset('model.nc', 'w')
        >>> Var = make_abc_testable(NetCDFVariableBase)
        >>> Var.subdevicenames = [
        ...     'element3', 'element1', 'element1_1', 'element2']
        >>> var = Var('flux_prec', isolate=True, timeaxis=1)
        >>> var.insert_subdevices(ncfile)
        >>> subdevice2index = var.query_subdevice2index(ncfile)
        >>> subdevice2index.get_index('element1_1')
        2
        >>> subdevice2index.get_index('element3')
        0
        >>> subdevice2index.get_index('element5')
        Traceback (most recent call last):
        ...
        OSError: No data for sequence `flux_prec` and (sub)device \
`element5` in NetCDF file `model.nc` available.

        Additionally, |NetCDFVariableBase.query_subdevice2index|
        checks for duplicates:

        >>> ncfile['station_id'][:] = str2chars(
        ...     ['element3', 'element1', 'element1_1', 'element1'])
        >>> var.query_subdevice2index(ncfile)
        Traceback (most recent call last):
        ...
        OSError: The NetCDF file `model.nc` contains duplicate (sub)device \
names for variable `flux_prec` (the first found duplicate is `element1`).

        >>> ncfile.close()
        """
        subdevices = self.query_subdevices(ncfile)
        self._test_duplicate_exists(ncfile, subdevices)
        subdev2index = {subdev: idx for (idx, subdev) in enumerate(subdevices)}
        return Subdevice2Index(subdev2index, self.name, get_filepath(ncfile))