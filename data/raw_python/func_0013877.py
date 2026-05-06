def _get_data_info(self, data, file_format):
        """Support file writing by determiniing data type and other options

        Parameters
        ----------
        data : pandas object
            Data to be written
        file_format : basestring
            String indicating netCDF3 or netCDF4

        Returns
        -------
        data_flag, datetime_flag, old_format
        """
        # get type of data
        data_type = data.dtype
        # check if older file_format
        # if file_format[:7] == 'NETCDF3':
        if file_format != 'NETCDF4':
            old_format = True
        else:
            old_format = False
        # check for object type
        if data_type != np.dtype('O'):
            # simple data, not an object
            
            # no 64bit ints in netCDF3
            if (data_type == np.int64) & old_format:
                data = data.astype(np.int32)
                data_type = np.int32
                   
            if data_type == np.dtype('<M8[ns]'):
                if not old_format:
                    data_type = np.int64
                else:
                    data_type = np.float
                datetime_flag = True
            else:
                datetime_flag = False
        else:
            # dealing with a more complicated object
            # iterate over elements until we hit something that is something,
            # and not NaN
            data_type = type(data.iloc[0])
            for i in np.arange(len(data)):
                if len(data.iloc[i]) > 0:
                    data_type = type(data.iloc[i])
                    if not isinstance(data_type, np.float):
                        break
            datetime_flag = False
            
        return data, data_type, datetime_flag