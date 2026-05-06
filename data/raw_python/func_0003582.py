def needle_positions_to_bytes(self, needle_positions):
        """Convert the needle positions to the wire format.

        This conversion is used for :ref:`cnfline`.

        :param needle_positions: an iterable over :attr:`needle_positions` of
          length :attr:`number_of_needles`
        :rtype: bytes
        """
        bit = self.needle_positions
        assert len(bit) == 2
        max_length = len(needle_positions)
        assert max_length == self.number_of_needles
        result = []
        for byte_index in range(0, max_length, 8):
            byte = 0
            for bit_index in range(8):
                index = byte_index + bit_index
                if index >= max_length:
                    break
                needle_position = needle_positions[index]
                if bit.index(needle_position) == 1:
                    byte |= 1 << bit_index
            result.append(byte)
            if byte_index >= max_length:
                break
        result.extend(b"\x00" * (25 - len(result)))
        return bytes(result)