def _insert_text_buf(self, line, idx):
        """Insert text into bytes buffers"""
        self._bytes_012[idx] = 0
        self._bytes_345[idx] = 0
        # Crop text if necessary
        I = np.array([ord(c) - 32 for c in line[:self._n_cols]])
        I = np.clip(I, 0, len(__font_6x8__)-1)
        if len(I) > 0:
            b = __font_6x8__[I]
            self._bytes_012[idx, :len(I)] = b[:, :3]
            self._bytes_345[idx, :len(I)] = b[:, 3:]