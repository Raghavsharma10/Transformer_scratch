def _table_to_str(self):
        # type: () -> str
        """Return single formatted table string.

        :return: str
        """
        _marker_line = self._marker_line()
        output = _marker_line + self._row_to_str(self.headers) + _marker_line

        for row in self._rows:
            output += self._row_to_str(row)

        output += _marker_line

        return output