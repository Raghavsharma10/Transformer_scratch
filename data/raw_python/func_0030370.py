def reopen(self):
      """Reopens the file. Usually used after it has been rotated."""
      # Read any remaining content in the file and store it in a buffer.
      self._read()
      # Close it to ensure we don't leak file descriptors
      self._close()
      
      # Reopen the file.
      try:
         self._open(self._path, skip_to_end = False)
         return True
      except OSError:
         # If opening fails, it was probably deleted.
         return False