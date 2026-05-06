def get_bytes(self, line_number):
        """Get the bytes representing needle positions or None.

        :param int line_number: the line number to take the bytes from
        :rtype: bytes
        :return: the bytes that represent the message or :obj:`None` if no
          data is there for the line.

        Depending on the :attr:`machine`, the length and result may vary.
        """
        if line_number not in self._needle_position_bytes_cache:
            line = self._get(line_number)
            if line is None:
                line_bytes = None
            else:
                line_bytes = self._machine.needle_positions_to_bytes(line)
            self._needle_position_bytes_cache[line_number] = line_bytes
        return self._needle_position_bytes_cache[line_number]