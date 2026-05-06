def readline(self, size=None):
        """
        Read one entire line from the file.  A trailing newline character is
        kept in the string (but may be absent when a file ends with an
        incomplete line).  If the size argument is present and non-negative, it
        is a maximum byte count (including the trailing newline) and an
        incomplete line may be returned.  An empty string is returned only when
        EOF is encountered immediately.

        @note: Unlike stdio's C{fgets()}, the returned string contains null
        characters (C{'\\0'}) if they occurred in the input.

        @param size: maximum length of returned string.
        @type size: int
        @return: next line of the file, or an empty string if the end of the
            file has been reached.
        @rtype: str
        """
        # it's almost silly how complex this function is.
        if self._closed:
            raise IOError('File is closed')
        if not (self._flags & self.FLAG_READ):
            raise IOError('File not open for reading')
        line = self._rbuffer
        while True:
            if self._at_trailing_cr and (self._flags & self.FLAG_UNIVERSAL_NEWLINE) and (len(line) > 0):
                # edge case: the newline may be '\r\n' and we may have read
                # only the first '\r' last time.
                if line[0] == '\n':
                    line = line[1:]
                    self._record_newline('\r\n')
                else:
                    self._record_newline('\r')
                self._at_trailing_cr = False
            # check size before looking for a linefeed, in case we already have
            # enough.
            if (size is not None) and (size >= 0):
                if len(line) >= size:
                    # truncate line and return
                    self._rbuffer = line[size:]
                    line = line[:size]
                    self._pos += len(line)
                    return line
                n = size - len(line)
            else:
                n = self._bufsize
            if ('\n' in line) or ((self._flags & self.FLAG_UNIVERSAL_NEWLINE) and ('\r' in line)):
                break
            try:
                new_data = self._read(n)
            except EOFError:
                new_data = None
            if (new_data is None) or (len(new_data) == 0):
                self._rbuffer = ''
                self._pos += len(line)
                return line
            line += new_data
            self._realpos += len(new_data)
        # find the newline
        pos = line.find('\n')
        if self._flags & self.FLAG_UNIVERSAL_NEWLINE:
            rpos = line.find('\r')
            if (rpos >= 0) and ((rpos < pos) or (pos < 0)):
                pos = rpos
        xpos = pos + 1
        if (line[pos] == '\r') and (xpos < len(line)) and (line[xpos] == '\n'):
            xpos += 1
        self._rbuffer = line[xpos:]
        lf = line[pos:xpos]
        line = line[:pos] + '\n'
        if (len(self._rbuffer) == 0) and (lf == '\r'):
            # we could read the line up to a '\r' and there could still be a
            # '\n' following that we read next time.  note that and eat it.
            self._at_trailing_cr = True
        else:
            self._record_newline(lf)
        self._pos += len(line)
        return line