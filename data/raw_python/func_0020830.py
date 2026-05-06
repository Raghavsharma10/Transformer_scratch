def i2c_master_read(self, addr, length, flags=I2C_NO_FLAGS):
        """Make an I2C read access.

        The given I2C device is addressed and clock cycles for `length` bytes
        are generated. A short read will occur if the device generates an early
        NAK.

        The transaction is finished with an I2C stop condition unless the
        I2C_NO_STOP flag is set.
        """

        data = array.array('B', (0,) * length)
        status, rx_len = api.py_aa_i2c_read_ext(self.handle, addr, flags,
                length, data)
        _raise_i2c_status_code_error_if_failure(status)
        del data[rx_len:]
        return bytes(data)