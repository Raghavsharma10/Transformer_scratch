def write_lines(self, lines, level=0):
        """Append multiple new lines"""

        for line in lines:
            self.write_line(line, level)