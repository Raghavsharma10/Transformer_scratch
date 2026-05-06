def _initApplicationList(self):
        """Query Asterisk Manager Interface to initialize internal list of 
        available applications.
        
        CLI Command - core show applications
        
        """
        if self.checkVersion('1.4'):
            cmd = "core show applications"
        else:
            cmd = "show applications"
        cmdresp = self.executeCommand(cmd)
        self._applications = set()
        for line in cmdresp.splitlines()[1:-1]:
            mobj = re.match('\s*(\S+):', line)
            if mobj:
                self._applications.add(mobj.group(1).lower())