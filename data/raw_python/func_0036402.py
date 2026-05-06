def load_buffer_six_to_ten(self, out_buffer):
        """
        Loads second program buffer (0x94) with everything but
        first three bytes and checksum
        """
        offset = 3
        for ind, step in enumerate(self.partial_steps_data(5)):
            struct.pack_into(b"< 2H", out_buffer, offset + ind*4, step[0], step[1])
        struct.pack_into(b"< B x", out_buffer, 23, self._program_mode)