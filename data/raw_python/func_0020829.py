def i2c_master_write(self, i2c_address, data, flags=I2C_NO_FLAGS):
        """Make an I2C write access.

        The given I2C device is addressed and data given as a string is
        written. The transaction is finished with an I2C stop condition unless
        I2C_NO_STOP is set in the flags.

        10 bit addresses are supported if the I2C_10_BIT_ADDR flag is set.
        """

        data = array.array('B', data)
        status, _ = api.py_aa_i2c_write_ext(self.handle, i2c_address, flags,
                len(data), data)
        _raise_i2c_status_code_error_if_failure(status)