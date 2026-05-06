def get_raw_data_from_buffer(self, filter_func=None, converter_func=None):
        '''Reads local data buffer and returns raw data array.

        Returns
        -------
        data : np.array
            An array containing data words from the local data buffer.
        '''
        if self._is_running:
            raise RuntimeError('Readout thread running')
        if not self.fill_buffer:
            logging.warning('Data buffer is not activated')
        return [convert_data_array(data_array_from_data_iterable(data_iterable), filter_func=filter_func, converter_func=converter_func) for data_iterable in self._data_buffer]