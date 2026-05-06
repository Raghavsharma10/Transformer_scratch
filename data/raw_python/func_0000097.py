def write_word(self, cmd, value):
        """
        Writes a 16-bit word to the specified command register
        """
        self.bus.write_word_data(self.address, cmd, value)
        self.log.debug(
            "write_word: Wrote 0x%04X to command register 0x%02X" % (
                value, cmd
            )
        )