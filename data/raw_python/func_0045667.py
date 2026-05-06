def insert(self, lines=None):
        """
        Insert lines into the editor.

        Note:
            To insert before the first line, use :func:`~exa.core.editor.Editor.preappend`
            (or key 0); to insert after the last line use :func:`~exa.core.editor.Editor.append`.

        Args:
            lines (dict): Dictionary of lines of form (lineno, string) pairs
        """
        for i, (key, line) in enumerate(lines.items()):
            n = key + i
            first_half = self._lines[:n]
            last_half = self._lines[n:]
            self._lines = first_half + [line] + last_half