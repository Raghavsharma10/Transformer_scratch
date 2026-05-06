def getCodecList(self):
        """Query Asterisk Manager Interface for defined codecs.
        
        CLI Command - core show codecs
        
        @return: Dictionary - Short Name -> (Type, Long Name)
        
        """
        if self.checkVersion('1.4'):
            cmd = "core show codecs"
        else:
            cmd = "show codecs"
        cmdresp = self.executeCommand(cmd)
        info_dict = {}
        for line in cmdresp.splitlines():
            mobj = re.match('\s*(\d+)\s+\((.+)\)\s+\((.+)\)\s+(\w+)\s+(\w+)\s+\((.+)\)$',
                            line)
            if mobj:
                info_dict[mobj.group(5)] = (mobj.group(4), mobj.group(6))
        return info_dict