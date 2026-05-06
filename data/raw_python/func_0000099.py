def write_block_data(self, cmd, block):
        """
        Writes a block of bytes to the bus using I2C format to the specified
        command register
        """
        self.bus.write_i2c_block_data(self.address, cmd, block)
        self.log.debug(
            "write_block_data: Wrote [%s] to command register 0x%02X" % (
                ', '.join(['0x%02X' % x for x in block]),
                cmd
            )
        )