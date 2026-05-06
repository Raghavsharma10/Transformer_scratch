def _sendStatCmd(self,  cmd):
        """Send stat command to Memcached Server and return response lines.
        
        @param cmd: Command string.
        @return:    Array of strings.
        
        """
        try:
            self._conn.write("%s\r\n" % cmd)
            regex = re.compile('^(END|ERROR)\r\n', re.MULTILINE)
            (idx, mobj, text) = self._conn.expect([regex,], self._timeout) #@UnusedVariable
        except:
            raise Exception("Communication with %s failed" % self._instanceName)
        if mobj is not None:
            if mobj.group(1) == 'END':
                return text.splitlines()[:-1]
            elif mobj.group(1) == 'ERROR':
                raise Exception("Protocol error in communication with %s."
                                % self._instanceName)
        else:
            raise Exception("Connection with %s timed out." % self._instanceName)