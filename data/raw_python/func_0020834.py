def i2c_slave_read(self):
        """Read the bytes from an I2C slave reception.

        The bytes are returned as a string object.
        """
        data = array.array('B', (0,) * self.BUFFER_SIZE)
        status, addr, rx_len = api.py_aa_i2c_slave_read_ext(self.handle,
                self.BUFFER_SIZE, data)
        _raise_i2c_status_code_error_if_failure(status)

        # In case of general call, actually return the general call address
        if addr == 0x80:
            addr = 0x00
        del data[rx_len:]
        return (addr, bytes(data))