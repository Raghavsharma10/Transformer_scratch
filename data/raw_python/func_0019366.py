def query_subdevices(self, ncfile) -> List[str]:
        """Query the names of the (sub)devices of the logged sequences
        from the given NetCDF file

        (1) We apply function |NetCDFVariableBase.query_subdevices| on
        an empty NetCDF file.  The error message shows that the method
        tries to query the (sub)device names both under the assumptions
        that variables have been isolated or not:

        >>> from hydpy.core.netcdftools import NetCDFVariableBase
        >>> from hydpy import make_abc_testable, TestIO
        >>> from hydpy.core.netcdftools import netcdf4
        >>> with TestIO():
        ...     ncfile = netcdf4.Dataset('model.nc', 'w')
        >>> Var = make_abc_testable(NetCDFVariableBase)
        >>> Var.subdevicenames = 'element1', 'element_2'
        >>> var = Var('flux_prec', isolate=False, timeaxis=1)
        >>> var.query_subdevices(ncfile)
        Traceback (most recent call last):
        ...
        OSError: NetCDF file `model.nc` does neither contain a variable \
named `flux_prec_station_id` nor `station_id` for defining the \
coordinate locations of variable `flux_prec`.

        (2) After inserting the (sub)device name, they can be queried
        and returned:

        >>> var.insert_subdevices(ncfile)
        >>> Var('flux_prec', isolate=False, timeaxis=1).query_subdevices(ncfile)
        ['element1', 'element_2']
        >>> Var('flux_prec', isolate=True, timeaxis=1).query_subdevices(ncfile)
        ['element1', 'element_2']

        >>> ncfile.close()
        """
        tests = ['%s%s' % (prefix, varmapping['subdevices'])
                 for prefix in ('%s_' % self.name, '')]
        for subdevices in tests:
            try:
                chars = ncfile[subdevices][:]
                break
            except (IndexError, KeyError):
                pass
        else:
            raise IOError(
                'NetCDF file `%s` does neither contain a variable '
                'named `%s` nor `%s` for defining the coordinate '
                'locations of variable `%s`.'
                % (get_filepath(ncfile), tests[0], tests[1], self.name))
        return chars2str(chars)