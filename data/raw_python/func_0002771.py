def in_file(self, fn: str) -> Iterator[InsertionPoint]:
        """
        Returns an iterator over all of the insertion points in a given file.
        """
        logger.debug("finding insertion points in file: %s", fn)
        yield from self.__file_insertions.get(fn, [])