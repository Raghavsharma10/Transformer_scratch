def get_line_configuration_message(self, line_number):
        """Return the cnfLine content without id for the line.

        :param int line_number: the number of the line
        :rtype: bytes
        :return: a cnfLine message without id as defined in :ref:`cnfLine`
        """
        if line_number not in self._line_configuration_message_cache:
            line_bytes = self.get_bytes(line_number)
            if line_bytes is not None:
                line_bytes = bytes([line_number & 255]) + line_bytes
                line_bytes += bytes([self.is_last(line_number)])
                line_bytes += crc8(line_bytes).digest()
            self._line_configuration_message_cache[line_number] = line_bytes
            del line_bytes
        line = self._line_configuration_message_cache[line_number]
        if line is None:
            # no need to cache a lot of empty lines
            line = (bytes([line_number & 255]) +
                    b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00' +
                    b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01')
            line += crc8(line).digest()
        return line