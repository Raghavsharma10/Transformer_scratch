def read_unsigned_word(self, cmd, little_endian=True):
        """
        Read an unsigned word from the specified command register
        We assume the data is in little endian mode, if it is in big endian
        mode then set little_endian to False
        """
        result = self.bus.read_word_data(self.address, cmd)

        if not little_endian:
            result = ((result << 8) & 0xFF00) + (result >> 8)

        self.log.debug(
            "read_unsigned_word: Read 0x%04X from command register 0x%02X" % (
                result, cmd
            )
        )
        return result