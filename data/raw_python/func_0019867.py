def _initModuleList(self):
        """Query Asterisk Manager Interface to initialize internal list of 
        loaded modules.
        
        CLI Command - core show modules
        
        """
        if self.checkVersion('1.4'):
            cmd = "module show"
        else:
            cmd = "show modules"
        cmdresp = self.executeCommand(cmd)
        self._modules = set()
        for line in cmdresp.splitlines()[1:-1]:
            mobj = re.match('\s*(\S+)\s', line)
            if mobj:
                self._modules.add(mobj.group(1).lower())