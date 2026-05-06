def _initAsteriskVersion(self):
        """Query Asterisk Manager Interface for Asterisk Version to configure
        system for compatibility with multiple versions
        .
        CLI Command - core show version

        """
        if self._ami_version > util.SoftwareVersion('1.0'):
            cmd = "core show version"
        else:
            cmd = "show version"
        cmdresp = self.executeCommand(cmd)
        mobj = re.match('Asterisk\s*(SVN-branch-|\s)(\d+(\.\d+)*)', cmdresp)
        if mobj:
            self._asterisk_version = util.SoftwareVersion(mobj.group(2))
        else:
            raise Exception('Asterisk version cannot be determined.')