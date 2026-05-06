def readline(self, timeout):
        """
        Read a line from the socket.  We assume no data is pending after the
        line, so it's okay to attempt large reads.
        """
        buf = self.__remainder
        while not '\n' in buf:
            buf += self._read_timeout(timeout)
        n = buf.index('\n')
        self.__remainder = buf[n+1:]
        buf = buf[:n]
        if (len(buf) > 0) and (buf[-1] == '\r'):
            buf = buf[:-1]
        return buf