def read_raw_data_from_fifo(self, fifo, filter_func=None, converter_func=None):
        '''Reads FIFO data and returns raw data array.

        Returns
        -------
        data : np.array
            An array containing FIFO data words.
        '''
        return convert_data_array(self.dut[fifo].get_data(), filter_func=filter_func, converter_func=converter_func)