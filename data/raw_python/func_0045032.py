def save(self, path=None, complevel=1, complib='zlib'):
        """
        Save the container as an HDF5 archive.

        Args:
            path (str): Path where to save the container
        """
        if path is None:
            path = self.hexuid + '.hdf5'
        elif os.path.isdir(path):
            path += os.sep + self.hexuid + '.hdf5'
        elif not (path.endswith('.hdf5') or path.endswith('.hdf')):
            raise ValueError('File path must have a ".hdf5" or ".hdf" extension.')
        with pd.HDFStore(path, 'w', complevel=complevel, complib=complib) as store:
            store['kwargs'] = pd.Series()
            store.get_storer('kwargs').attrs.metadata = self._rel()
            fc = 0    # Field counter (see special handling of fields below)
            for name, data in self._data().items():
                if hasattr(data, '_revert_categories'):
                    data._revert_categories()
                name = name[1:] if name.startswith('_') else name
                if isinstance(data, Field):    # Fields are handled separately
                    fname = 'FIELD{}_'.format(fc) + name + '/'
                    store[fname + 'data'] = pd.DataFrame(data)
                    for i, field in enumerate(data.field_values):
                        ffname = fname + 'values' + str(i)
                        if isinstance(field, pd.Series):
                            store[ffname] = pd.Series(field)
                        else:
                            store[ffname] = pd.DataFrame(field)
                    fc += 1
                elif isinstance(data, Series):
                    s = pd.Series(data)
                    if isinstance(data.dtype, pd.types.dtypes.CategoricalDtype):
                        s = s.astype('O')
                    store[name] = s
                elif isinstance(data, DataFrame):
                    store[name] = pd.DataFrame(data)
                elif isinstance(data, SparseSeries):
                    s = pd.SparseSeries(data)
                    if isinstance(data.dtype, pd.types.dtypes.CategoricalDtype):
                        s = s.astype('O')
                    store[name] = s
                elif isinstance(data, SparseDataFrame):
                    store[name] = pd.SparseDataFrame(data)
                else:
                    if hasattr(data, 'dtype') and isinstance(data.dtype, pd.types.dtypes.CategoricalDtype):
                        data = data.astype('O')
                    else:
                        for col in data:
                            if isinstance(data[col].dtype, pd.types.dtypes.CategoricalDtype):
                                data[col] = data[col].astype('O')
                    store[name] = data
                if hasattr(data, '_set_categories'):
                    data._set_categories()