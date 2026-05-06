def line_is_interesting(self, line):
        """Return True, False, or None.

        True means always output, False means never output, None means output
        only if there are interesting lines.

        """
        if line.startswith('Name'):
            return None
        if line.startswith('--------'):
            return None
        if line.startswith('TOTAL'):
            return None
        if '100%' in line:
            return False
        if line == '\n':
            return None if self._last_line_was_printable else False
        return True