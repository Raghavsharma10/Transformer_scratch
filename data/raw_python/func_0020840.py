def spi_write(self, data):
        """Write a stream of bytes to a SPI device."""
        data_out = array.array('B', data)
        data_in = array.array('B', (0,) * len(data_out))
        ret = api.py_aa_spi_write(self.handle, len(data_out), data_out,
                len(data_in), data_in)
        _raise_error_if_negative(ret)
        return bytes(data_in)