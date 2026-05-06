def decipher_block (self, state):
        """Perform AES block decipher on input"""
        if len(state) != 16:
            Log.error(u"Expecting block of 16")

        self._add_round_key(state, self._Nr)

        for i in range(self._Nr - 1, 0, -1):
            self._i_shift_rows(state)
            self._i_sub_bytes(state)
            self._add_round_key(state, i)
            self._mix_columns(state, True)

        self._i_shift_rows(state)
        self._i_sub_bytes(state)
        self._add_round_key(state, 0)
        return state