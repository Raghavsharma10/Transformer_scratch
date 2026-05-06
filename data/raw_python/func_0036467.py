def col_widths(self):
        # type: () -> defaultdict
        """Get MAX possible width of each column in the table.

        :return: defaultdict
        """
        _widths = defaultdict(int)

        all_rows = [self.headers]
        all_rows.extend(self._rows)

        for row in all_rows:
            for idx, col in enumerate(row):
                _col_l = len(col)
                if _col_l > _widths[idx]:
                    _widths[idx] = _col_l

        return _widths