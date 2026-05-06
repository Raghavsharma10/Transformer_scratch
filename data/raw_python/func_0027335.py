def name(self):
        """the instrument's name (5 characters, zero-padded)"""
        instr_name = self.song.song_data.instrument_names[self.index]

        if type(instr_name) == bytes:
            instr_name = instr_name.decode('utf-8')

        return instr_name