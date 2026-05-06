def spi_configure(self, polarity, phase, bitorder):
        """Configure the SPI interface."""
        ret = api.py_aa_spi_configure(self.handle, polarity, phase, bitorder)
        _raise_error_if_negative(ret)