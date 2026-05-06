def read_block_data(self, cmd, length):
        """
        Read a block of bytes from the bus from the specified command register
        Amount of bytes read in is defined by length
        """
        results = self.bus.read_i2c_block_data(self.address, cmd, length)
        self.log.debug(
            "read_block_data: Read [%s] from command register 0x%02X" % (
                ', '.join(['0x%02X' % x for x in results]),
                cmd
            )
        )
        return results