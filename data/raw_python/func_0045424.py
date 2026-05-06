def makefile(self, mode='r', bufsize=-1):
        """
        Returns a special fileobject that has corutines instead of the usual
        read/readline/write methods. Will work in the same manner though.
        """
        return _fileobject(Socket(
            _sock=self._fd._sock, 
            _timeout=self._timeout, 
            _proactor_added=self._proactor_added
        ), mode, bufsize)