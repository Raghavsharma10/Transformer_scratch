def _marker_line(self):
        # type: () -> str
        """Generate a correctly sized marker line.

        e.g.

        '+------------------+---------+----------+---------+'

        :return: str
        """
        output = ''
        for col in sorted(self.col_widths):
            line = self.COLUMN_MARK + (self.DASH * (self.col_widths[col] + self.PADDING * 2))
            output += line
        output += self.COLUMN_MARK + '\n'

        return output