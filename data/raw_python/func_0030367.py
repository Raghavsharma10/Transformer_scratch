def _open(self, path, skip_to_end = True, offset = None):
      """Open `path`, optionally seeking to the end if `skip_to_end` is True."""
      fh = os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK))

      # If the file is being opened for the first time, jump to the end.
      # Otherwise, it is being reopened after a rotation, and we want
      # content from the beginning.
      if offset is None:
         if skip_to_end:
            fh.seek(0, 2)

            self._offset = fh.tell()
         else:
            self._offset = 0
      else:
         fh.seek(offset)
         self._offset = fh.tell()
      
      self._fh = fh
      self._lastsize = fh.tell()
      self._inode = os.stat(self._path).st_ino