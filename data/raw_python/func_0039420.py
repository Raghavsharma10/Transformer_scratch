def _root_dirent(self):
        """Returns the root folder dirent as filesystem_walk API doesn't."""
        fstat = self._filesystem.stat('/')

        yield Dirent(fstat['ino'], self._filesystem.path('/'),
                     fstat['size'], 'd', True,
                     timestamp(fstat['atime'], 0),
                     timestamp(fstat['mtime'], 0),
                     timestamp(fstat['ctime'], 0),
                     0)