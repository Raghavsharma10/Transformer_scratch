def _set_mode(self, mode='r', bufsize=-1):
        """
        Subclasses call this method to initialize the BufferedFile.
        """
        # set bufsize in any event, because it's used for readline().
        self._bufsize = self._DEFAULT_BUFSIZE
        if bufsize < 0:
            # do no buffering by default, because otherwise writes will get
            # buffered in a way that will probably confuse people.
            bufsize = 0
        if bufsize == 1:
            # apparently, line buffering only affects writes.  reads are only
            # buffered if you call readline (directly or indirectly: iterating
            # over a file will indirectly call readline).
            self._flags |= self.FLAG_BUFFERED | self.FLAG_LINE_BUFFERED
        elif bufsize > 1:
            self._bufsize = bufsize
            self._flags |= self.FLAG_BUFFERED
            self._flags &= ~self.FLAG_LINE_BUFFERED
        elif bufsize == 0:
            # unbuffered
            self._flags &= ~(self.FLAG_BUFFERED | self.FLAG_LINE_BUFFERED)

        if ('r' in mode) or ('+' in mode):
            self._flags |= self.FLAG_READ
        if ('w' in mode) or ('+' in mode):
            self._flags |= self.FLAG_WRITE
        if ('a' in mode):
            self._flags |= self.FLAG_WRITE | self.FLAG_APPEND
            self._size = self._get_size()
            self._pos = self._realpos = self._size
        if ('b' in mode):
            self._flags |= self.FLAG_BINARY
        if ('U' in mode):
            self._flags |= self.FLAG_UNIVERSAL_NEWLINE
            # built-in file objects have this attribute to store which kinds of
            # line terminations they've seen:
            # <http://www.python.org/doc/current/lib/built-in-funcs.html>
            self.newlines = None