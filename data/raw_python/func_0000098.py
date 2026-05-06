def write_raw_byte(self, value):
        """
        Writes an 8-bit byte directly to the bus
        """
        self.bus.write_byte(self.address, value)
        self.log.debug("write_raw_byte: Wrote 0x%02X" % value)