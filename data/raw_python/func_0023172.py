def _do_pending_writes(self):
        """Do any pending text writes"""
        for text, wrap in self._pending_writes:
            # truncate in case of *really* long messages
            text = text[-self._n_cols*self._n_rows:]
            text = text.split('\n')
            text = [t if len(t) > 0 else '' for t in text]
            nr, nc = self._n_rows, self._n_cols
            for para in text:
                para = para[:nc] if not wrap else para
                lines = [para[ii:(ii+nc)] for ii in range(0, len(para), nc)]
                lines = [''] if len(lines) == 0 else lines
                for line in lines:
                    # Update row and scroll if necessary
                    self._text_lines.insert(0, line)
                    self._text_lines = self._text_lines[:nr]
                    self._bytes_012[1:] = self._bytes_012[:-1]
                    self._bytes_345[1:] = self._bytes_345[:-1]
                    self._insert_text_buf(line, 0)
        self._pending_writes = []