def i2c_monitor_read(self):
        """Retrieved any data fetched by the monitor.

        This function has an integrated timeout mechanism. You should use
        :func:`poll` to determine if there is any data available.

        Returns a list of data bytes and special symbols. There are three
        special symbols: `I2C_MONITOR_NACK`, I2C_MONITOR_START and
        I2C_MONITOR_STOP.

        """
        data = array.array('H', (0,) * self.BUFFER_SIZE)
        ret = api.py_aa_i2c_monitor_read(self.handle, self.BUFFER_SIZE,
                data)
        _raise_error_if_negative(ret)
        del data[ret:]
        return data.tolist()