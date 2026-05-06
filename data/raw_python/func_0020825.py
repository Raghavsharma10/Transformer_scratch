def i2c_bitrate(self):
        """I2C bitrate in kHz. Not every bitrate is supported by the host
        adapter. Therefore, the actual bitrate may be less than the value which
        is set.

        The power-on default value is 100 kHz.
        """

        ret = api.py_aa_i2c_bitrate(self.handle, 0)
        _raise_error_if_negative(ret)
        return ret