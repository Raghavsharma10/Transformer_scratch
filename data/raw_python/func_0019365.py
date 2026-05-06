def insert_subdevices(self, ncfile) -> None:
        """Insert a variable of the names of the (sub)devices of the logged
        sequences into the given NetCDF file

        (1) We prepare a |NetCDFVariableBase| subclass with fixed
        (sub)device names:

        >>> from hydpy.core.netcdftools import NetCDFVariableBase, chars2str
        >>> from hydpy import make_abc_testable, TestIO
        >>> from hydpy.core.netcdftools import netcdf4
        >>> Var = make_abc_testable(NetCDFVariableBase)
        >>> Var.subdevicenames = 'element1', 'element_2'

        (2) Without isolating variables,
        |NetCDFVariableBase.insert_subdevices| prefixes the name of the
        |NetCDFVariableBase| object to the name of the inserted variable
        and its dimensions.  The first dimension corresponds to the
        number of (sub)devices, the second dimension to the number of
        characters of the longest (sub)device name:

        >>> var1 = Var('var1', isolate=False, timeaxis=1)
        >>> with TestIO():
        ...     file1 = netcdf4.Dataset('model1.nc', 'w')
        >>> var1.insert_subdevices(file1)
        >>> file1['var1_station_id'].dimensions
        ('var1_stations', 'var1_char_leng_name')
        >>> file1['var1_station_id'].shape
        (2, 9)
        >>> chars2str(file1['var1_station_id'][:])
        ['element1', 'element_2']
        >>> file1.close()

        (3) When isolating variables, we omit the prefix:

        >>> var2 = Var('var2', isolate=True, timeaxis=1)
        >>> with TestIO():
        ...     file2 = netcdf4.Dataset('model2.nc', 'w')
        >>> var2.insert_subdevices(file2)
        >>> file2['station_id'].dimensions
        ('stations', 'char_leng_name')
        >>> file2['station_id'].shape
        (2, 9)
        >>> chars2str(file2['station_id'][:])
        ['element1', 'element_2']
        >>> file2.close()
        """
        prefix = self.prefix
        nmb_subdevices = '%s%s' % (prefix, dimmapping['nmb_subdevices'])
        nmb_characters = '%s%s' % (prefix, dimmapping['nmb_characters'])
        subdevices = '%s%s' % (prefix, varmapping['subdevices'])
        statchars = str2chars(self.subdevicenames)
        create_dimension(ncfile, nmb_subdevices, statchars.shape[0])
        create_dimension(ncfile, nmb_characters, statchars.shape[1])
        create_variable(
            ncfile, subdevices, 'S1', (nmb_subdevices, nmb_characters))
        ncfile[subdevices][:, :] = statchars