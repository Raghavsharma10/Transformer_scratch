def nodes(self, path):
        """Iterates over the files and directories contained within the disk
        starting from the given path.

        Yields the path of the nodes.

        """
        path = posix_path(path)

        yield from (self.path(path, e) for e in self._handler.find(path))