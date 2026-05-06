def readlines(self):
      """A generator producing lines from the file."""

      # If the file is not open, there's nothing to return
      if not self._fh:
         raise StopIteration

      at_eof = False
      while True:
         # Clean the buffer sometimes.
         if self._bufoffset > (self._maxreadsize / 2):
            self._buf = self._buf[self._bufoffset:]
            self._bufoffset = 0

         # Fill up the buffer if necessary.
         if len(self._buf) < self._maxreadsize:
            at_eof = not self._read(self._maxreadsize)

         # Look for the next line.
         try:
            next_newline = self._buf.index("\n", self._bufoffset)
            line = self._buf[self._bufoffset:next_newline]
            self._bufoffset = next_newline + 1
            # Save the current file offset for yielding and advance the file offset.
            offset = self._offset
            self._offset += len(line) + 1
            if self._longline:
               # This is the remaining chunk of a long line, we're not going
               # to yield it.
               self._longline = False
            else:
               yield line, offset

         except ValueError:
            # Reached the end of the buffer without finding any newlines.
            if not at_eof:
               # Line is longer than the half the buffer size? - Nope
               logger.warning("Skipping over longline at %s:%d", self._path,
                                                                 self._offset)
               self._bufoffset = len(self._buf) - 1
               self._longline = True
            raise StopIteration