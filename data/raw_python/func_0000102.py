def read_unsigned_byte(self, cmd):
        """
        Read an unsigned byte from the specified command register
        """
        result = self.bus.read_byte_data(self.address, cmd)
        self.log.debug(
            "read_unsigned_byte: Read 0x%02X from command register 0x%02X" % (
                result, cmd
            )
        )
        return result