def remove(self, member):
        """Remove a member from the archive."""
        # Make sure we have an info object
        if isinstance(member, ZipInfo):
            # 'member' is already an info object
            zinfo = member
        else:
            # Get info object for name
            zinfo = self.getinfo(member)

        # compute the location of the file data in the local file header,
        # by adding the lengths of the records before it
        zlen = len(zinfo.FileHeader()) + zinfo.compress_size
        fileidx = self.filelist.index(zinfo)
        fileofs = sum(
            [len(self.filelist[f].FileHeader()) + self.filelist[f].compress_size
            for f in xrange(0, fileidx)]
            )

        self.fp.seek(fileofs + zlen)
        after = self.fp.read()
        self.fp.seek(fileofs)
        self.fp.write(after)
        self.fp.seek(-zlen, 2)
        self.fp.truncate()

        self._didModify = True
        self.filelist.remove(zinfo)
        del self.NameToInfo[member]