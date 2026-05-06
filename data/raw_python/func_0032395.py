def reset(self):
        """
        In addition to the behavior of the superclass, delete any dangling
        lockfiles which may prevent this index from being opened.  With the
        tested version of PyLucene (something pre-2.0), this appears to not
        actually be necessary: deleting the entire index directory but
        leaving the lockfile in place seems to still allow the index to be
        recreated (perhaps because when the directory does not exist, we
        pass True as the create flag when opening the FSDirectory, I am
        uncertain).  Nevertheless, do this anyway for now.
        """
        RemoteIndexer.reset(self)
        if hasattr(self, '_lockfile'):
            os.remove(self._lockfile)
            del self._lockfile