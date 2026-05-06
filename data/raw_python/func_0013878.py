def _filter_netcdf4_metadata(self, mdata_dict, coltype, remove=False):
        """Filter metadata properties to be consistent with netCDF4.
        
        Notes
        -----
        removed forced to True if coltype consistent with a string type
        
        Parameters
        ----------
        mdata_dict : dict
            Dictionary equivalent to Meta object info
        coltype : type
            Type provided by _get_data_info
        remove : boolean (False)
            Removes FillValue and associated parameters disallowed for strings
            
        Returns
        -------
        dict
            Modified as needed for netCDf4
        
        """
        # Coerce boolean types to integers
        for key in mdata_dict:
            if type(mdata_dict[key]) == bool:
                mdata_dict[key] = int(mdata_dict[key])
        if (coltype == type(' ')) or (coltype == type(u' ')):
            remove = True
        # print ('coltype', coltype, remove, type(coltype), )
        if u'_FillValue' in mdata_dict.keys():
            # make sure _FillValue is the same type as the data
            if remove:
                mdata_dict.pop('_FillValue')
            else:
                mdata_dict['_FillValue'] = np.array(mdata_dict['_FillValue']).astype(coltype)
        if u'FillVal' in mdata_dict.keys():
            # make sure _FillValue is the same type as the data
            if remove:
                mdata_dict.pop('FillVal')
            else:
                mdata_dict['FillVal'] = np.array(mdata_dict['FillVal']).astype(coltype)
        return mdata_dict