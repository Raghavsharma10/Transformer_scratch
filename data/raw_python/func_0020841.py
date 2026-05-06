def spi_ss_polarity(self, polarity):
        """Change the ouput polarity on the SS line.

        Please note, that this only affects the master functions.
        """
        ret = api.py_aa_spi_master_ss_polarity(self.handle, polarity)
        _raise_error_if_negative(ret)