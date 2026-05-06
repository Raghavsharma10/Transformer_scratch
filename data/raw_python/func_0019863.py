def _getGreeting(self):
        """Read and parse Asterisk Manager Interface Greeting to determine and
        set Manager Interface version.

        """
        greeting = self._conn.read_until("\r\n", connTimeout)
        mobj = re.match('Asterisk Call Manager\/([\d\.]+)\s*$', greeting)
        if mobj:
            self._ami_version = util.SoftwareVersion(mobj.group(1))
        else:
            raise Exception("Asterisk Manager Interface version cannot be determined.")