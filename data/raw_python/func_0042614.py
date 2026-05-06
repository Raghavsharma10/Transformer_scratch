def cipher_block (self, state):
        """Perform AES block cipher on input"""
        # PKCS7 Padding
        state=state+[16-len(state)]*(16-len(state))# Fails test if it changes the input with +=

        self._add_round_key(state, 0)

        for i in range(1, self._Nr):
            self._sub_bytes(state)
            self._shift_rows(state)
            self._mix_columns(state, False)
            self._add_round_key(state, i)

        self._sub_bytes(state)
        self._shift_rows(state)
        self._add_round_key(state, self._Nr)
        return state