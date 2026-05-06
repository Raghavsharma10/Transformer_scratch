async def seek(self, pos, whence=sync_io.SEEK_SET):
        """Move to new file position.

        Argument offset is a byte count.  Optional argument whence defaults to
        SEEK_SET or 0 (offset from start of file, offset should be >= 0); other
        values are SEEK_CUR or 1 (move relative to current position, positive
        or negative), and SEEK_END or 2 (move relative to end of file, usually
        negative, although many platforms allow seeking beyond the end of a
        file).

        Note that not all file objects are seekable.
        """
        return self._stream.seek(pos, whence)