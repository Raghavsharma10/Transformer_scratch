def i2c_pullups(self):
        """Setting this to `True` will enable the I2C pullup resistors. If set
        to `False` the pullup resistors will be disabled.

        Raises an :exc:`IOError` if the hardware adapter does not support
        pullup resistors.
        """
        ret = api.py_aa_i2c_pullup(self.handle, I2C_PULLUP_QUERY)
        _raise_error_if_negative(ret)
        return ret