def write_byte(self, cmd, value):
        """
        Writes an 8-bit byte to the specified command register
        """
        self.bus.write_byte_data(self.address, cmd, value)
        self.log.debug(
            "write_byte: Wrote 0x%02X to command register 0x%02X" % (
                value, cmd
            )
        )