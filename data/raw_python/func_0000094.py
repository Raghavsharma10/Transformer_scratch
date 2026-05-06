def clean_up(self):
        """
        Close the I2C bus
        """
        self.log.debug("Closing I2C bus for address: 0x%02X" % self.address)
        self.bus.close()