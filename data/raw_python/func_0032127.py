def wasModified(self):
        """
        Check to see if this module has been modified on disk since the last
        time it was cached.

        @return: True if it has been modified, False if not.
        """
        self.filePath.restat()
        mtime = self.filePath.getmtime()
        if mtime >= self.lastModified:
            return True
        else:
            return False