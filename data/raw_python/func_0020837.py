def spi_bitrate(self):
        """SPI bitrate in kHz. Not every bitrate is supported by the host
        adapter. Therefore, the actual bitrate may be less than the value which
        is set. The slowest bitrate supported is 125kHz. Any smaller value will
        be rounded up to 125kHz.

        The power-on default value is 1000 kHz.
        """
        ret = api.py_aa_spi_bitrate(self.handle, 0)
        _raise_error_if_negative(ret)
        return ret