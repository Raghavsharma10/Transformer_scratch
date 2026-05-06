def in_file(self, fn: str) -> Iterator[Statement]:
        """
        Returns an iterator over all of the statements belonging to a file.
        """
        yield from self.__file_to_statements.get(fn, [])