def enable_i2c_slave(self, slave_address):
        """Enable I2C slave mode.

        The device will respond to the specified slave_address if it is
        addressed.

        You can wait for the data with :func:`poll` and get it with
        `i2c_slave_read`.
        """
        ret = api.py_aa_i2c_slave_enable(self.handle, slave_address,
                self.BUFFER_SIZE, self.BUFFER_SIZE)
        _raise_error_if_negative(ret)