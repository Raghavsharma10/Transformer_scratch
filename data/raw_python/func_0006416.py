def get_data_from_buffer(self, filter_func=None, converter_func=None):
        '''Reads local data buffer and returns data and meta data list.

        Returns
        -------
        data : list
            List of data and meta data dicts.
        '''
        if self._is_running:
            raise RuntimeError('Readout thread running')
        if not self.fill_buffer:
            logging.warning('Data buffer is not activated')
        return [convert_data_iterable(data_iterable, filter_func=filter_func, converter_func=converter_func) for data_iterable in self._data_buffer]