def i2c_bus_timeout(self):
        """I2C bus lock timeout in ms.

        Minimum value is 10 ms and the maximum value is 450 ms. Not every value
        can be set and will be rounded to the next possible number. You can
        read back the property to get the actual value.

        The power-on default value is 200 ms.
        """
        ret = api.py_aa_i2c_bus_timeout(self.handle, 0)
        _raise_error_if_negative(ret)
        return ret