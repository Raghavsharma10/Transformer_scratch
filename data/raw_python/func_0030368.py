def _read(self, limit = None):
      """Checks the file for new data and refills the buffer if it finds any."""
      # The code that used to be here was self._fh.read(limit)
      # However, this broke on OSX. os.read, however, works fine, but doesn't
      # take the None argument or have any way to specify "read to the end".
      # This emulates that behaviour.
      while True:
         # Check that we haven't closed this file
         if not self._fh:
            return False
         dataread = os.read(self._fh.fileno(), limit or 65535)
         if len(dataread) > 0:
            self._buf += dataread
            if limit is not None:
               return True
         else:
            return False