def i2c_master_write_read(self, i2c_address, data, length):
        """Make an I2C write/read access.

        First an I2C write access is issued. No stop condition will be
        generated. Instead the read access begins with a repeated start.

        This method is useful for accessing most addressable I2C devices like
        EEPROMs, port expander, etc.

        Basically, this is just a convenient function which internally uses
        `i2c_master_write` and `i2c_master_read`.
        """

        self.i2c_master_write(i2c_address, data, I2C_NO_STOP)
        return self.i2c_master_read(i2c_address, length)