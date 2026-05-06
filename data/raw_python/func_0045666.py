def append(self, lines):
        """
        Args:
            lines (list): List of line strings to append to the end of the editor
        """
        if isinstance(lines, list):
            self._lines = self._lines + lines
        elif isinstance(lines, str):
            lines = lines.split('\n')
            self._lines = self._lines + lines
        else:
            raise TypeError('Unsupported type {0} for lines.'.format(type(lines)))