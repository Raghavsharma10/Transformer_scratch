def status(self):
        """See :py:meth:`~stash.repository.Repository.status`."""
        result = set()
        for line in self._execute('svn stat')[1].splitlines():
            if line[0] == '?':
                result.add((FileStatus.Added, line[2:].strip()))
            elif line[0] == '!':
                result.add((FileStatus.Removed, line[2:].strip()))
        return result