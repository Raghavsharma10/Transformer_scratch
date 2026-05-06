def _initChannelTypesList(self):
        """Query Asterisk Manager Interface to initialize internal list of 
        supported channel types.
        
        CLI Command - core show applications
        
        """
        if self.checkVersion('1.4'):
            cmd = "core show channeltypes"
        else:
            cmd = "show channeltypes"
        cmdresp = self.executeCommand(cmd)
        self._chantypes = set()
        for line in cmdresp.splitlines()[2:]:
            mobj = re.match('\s*(\S+)\s+.*\s+(yes|no)\s+', line)
            if mobj:
                self._chantypes.add(mobj.group(1).lower())