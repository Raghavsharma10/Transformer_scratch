def newFile(self, *path):
        """
        Open a new file somewhere in this Store's file area.

        @param path: a sequence of path segments.

        @return: an L{AtomicFile}.
        """
        assert len(path) > 0, "newFile requires a nonzero number of segments"
        if self.dbdir is None:
            if self.filesdir is None:
                raise RuntimeError("This in-memory store has no file directory")
            else:
                tmpbase = self.filesdir
        else:
            tmpbase = self.dbdir
        tmpname = tmpbase.child('temp').child(str(tempCounter.next()) + ".tmp")
        return AtomicFile(tmpname.path, self.newFilePath(*path))