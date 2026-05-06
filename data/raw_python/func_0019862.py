def _getResponse(self):
        """Read and parse response from Asterisk Manager Interface.
        
        @return: Dictionary with response key-value pairs.

        """
        resp_dict= dict()
        resp_str = self._conn.read_until("\r\n\r\n", connTimeout)
        for line in resp_str.split("\r\n"):
            mobj = re.match('(\w+):\s*(\S.*)$', line);
            if mobj:
                resp_dict[mobj.group(1)] = mobj.group(2)
            else:
                mobj = re.match('(.*)--END COMMAND--\s*$', line, flags=re.DOTALL)
                if mobj:
                    resp_dict['command_response'] = mobj.group(1)
        return resp_dict