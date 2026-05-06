def get_curr_lines(self):
        """Return the current line number in the template,
        as well as the surrounding source lines
        """
        start = max(0, self._coord.line - 5)
        end = min(len(self._template_lines), self._coord.line + 4)

        lines = [(x, self._template_lines[x]) for x in six.moves.range(start, end, 1)]
        return self._coord.line, lines