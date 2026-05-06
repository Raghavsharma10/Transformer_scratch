def write_uint16(self, word):
        """Write 2 bytes."""
        self.write_byte(nyamuk_net.MOSQ_MSB(word))
        self.write_byte(nyamuk_net.MOSQ_LSB(word))