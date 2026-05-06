def write_quick(self):
        """
        Send only the read / write bit
        """
        self.bus.write_quick(self.address)
        self.log.debug("write_quick: Sent the read / write bit")