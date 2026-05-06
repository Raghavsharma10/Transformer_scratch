def at_line(self, line: FileLine) -> Iterator[InsertionPoint]:
        """
        Returns an iterator over all of the insertion points located at a
        given line.
        """
        logger.debug("finding insertion points at line: %s", str(line))
        filename = line.filename  # type: str
        line_num = line.num  # type: int
        for ins in self.in_file(filename):
            if line_num == ins.location.line:
                logger.debug("found insertion point at line [%s]: %s",
                             str(line), ins)
                yield ins