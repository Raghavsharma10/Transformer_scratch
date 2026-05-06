def _row_to_str(self, row):
        # type: (List[str]) -> str
        """Converts a list of strings to a correctly spaced and formatted
        row string.

        e.g.

        ['some', 'foo', 'bar'] --> '| some | foo |  bar  |'

        :param row: list
        :return: str
        """
        _row_text = ''
        for col, width in self.col_widths.items():
            _row_text += self.COLUMN_SEP
            l_pad, r_pad = self._split_int(width - len(row[col]))
            _row_text += '{0}{1}{2}'.format(' ' * (l_pad + self.PADDING),
                                            row[col],
                                            ' ' * (r_pad + self.PADDING))

        _row_text += self.COLUMN_SEP + '\n'

        return _row_text