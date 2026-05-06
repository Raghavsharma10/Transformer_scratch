def cmd_print(self):
        """Returns the raw lines to be printed."""
        if not self._valid_lines:
            return ''
        return '\n'.join([line.raw_line for line in self._valid_lines]) + '\n'