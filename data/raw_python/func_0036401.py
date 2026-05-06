def load_buffer_one_to_five(self, out_buffer):
        """
        Loads first program buffer (0x93) with everything but
        first three bytes and checksum
        """
        struct.pack_into(b"< 2B", out_buffer, 3, self._program_type, len(self._prog_steps))
        offset = 5
        for ind, step in enumerate(self.partial_steps_data(0)):
            struct.pack_into(b"< 2H", out_buffer, offset + ind*4, step[0], step[1])