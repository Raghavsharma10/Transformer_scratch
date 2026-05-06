def getFaxStatsCounters(self):
        """Query Asterisk Manager Interface for Fax Stats.
        
        CLI Command - fax show stats
        
        @return: Dictionary of fax stats.
        
        """
        if not self.hasFax():
            return None
        info_dict = {}
        cmdresp = self.executeCommand('fax show stats')
        ctxt = 'general'
        for section in cmdresp.strip().split('\n\n')[1:]:
            i = 0
            for line in section.splitlines():
                mobj = re.match('(\S.*\S)\s*:\s*(\d+)\s*$', line)
                if mobj:
                    if not info_dict.has_key(ctxt):
                        info_dict[ctxt] = {}
                    info_dict[ctxt][mobj.group(1).lower()] = int(mobj.group(2).lower())
                elif i == 0:
                    ctxt = line.strip().lower()
                i += 1    
        return info_dict