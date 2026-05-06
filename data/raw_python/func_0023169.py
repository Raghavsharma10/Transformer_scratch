def _resize_buffers(self, font_scale):
        """Resize buffers only if necessary"""
        new_sizes = (font_scale,) + self.size
        if new_sizes == self._current_sizes:  # don't need resize
            return
        self._n_rows = int(max(self.size[1] /
                               (self._char_height * font_scale), 1))
        self._n_cols = int(max(self.size[0] /
                               (self._char_width * font_scale), 1))
        self._bytes_012 = np.zeros((self._n_rows, self._n_cols, 3), np.float32)
        self._bytes_345 = np.zeros((self._n_rows, self._n_cols, 3), np.float32)
        pos = np.empty((self._n_rows, self._n_cols, 2), np.float32)
        C, R = np.meshgrid(np.arange(self._n_cols), np.arange(self._n_rows))
        # We are in left, top orientation
        x_off = 4.
        y_off = 4 - self.size[1] / font_scale
        pos[..., 0] = x_off + self._char_width * C
        pos[..., 1] = y_off + self._char_height * R
        self._position = VertexBuffer(pos)

        # Restore lines
        for ii, line in enumerate(self._text_lines[:self._n_rows]):
            self._insert_text_buf(line, ii)
        self._current_sizes = new_sizes