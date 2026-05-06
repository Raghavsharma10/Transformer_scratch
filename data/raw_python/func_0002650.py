def at_line(self, line: FileLine) -> Iterator[Statement]:
        """
        Returns an iterator over all of the statements located at a given line.
        """
        num = line.num
        for stmt in self.in_file(line.filename):
            if stmt.location.start.line == num:
                yield stmt