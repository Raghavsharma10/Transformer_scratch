def group(self, indent: int = DEFAULT_INDENT, add_line: bool = True) -> _TextGroup:
        """
        Returns a context manager which adds an indentation before each line.

        :param indent: Number of spaces to print.
        :param add_line: If True, a new line will be printed after the group.
        :return: A TextGroup context manager.
        """
        return _TextGroup(self, indent, add_line)