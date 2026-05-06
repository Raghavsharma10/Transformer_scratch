def read_raw_byte(self):
        """
        Read an 8-bit byte directly from the bus
        """
        result = self.bus.read_byte(self.address)
        self.log.debug("read_raw_byte: Read 0x%02X from the bus" % result)
        return result