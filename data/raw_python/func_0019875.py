def getPeerStats(self, chantype):
        """Query Asterisk Manager Interface for SIP / IAX2 Peer Stats.
        
        CLI Command - sip show peers
                      iax2 show peers
        
        @param chantype: Must be 'sip' or 'iax2'.
        @return:         Dictionary of statistics counters for VoIP Peers.

        """
        chan = chantype.lower()
        if not self.hasChannelType(chan):
            return None
        if chan == 'iax2':
            cmd = "iax2 show peers"
        elif chan == 'sip':
            cmd = "sip show peers"
        else:
            raise AttributeError("Invalid channel type in query for Peer Stats.")
        cmdresp = self.executeCommand(cmd)
        
        info_dict = dict(
            online = 0, unreachable = 0, lagged = 0, 
            unknown = 0, unmonitored = 0)
        for line in cmdresp.splitlines():
            if re.search('ok\s+\(\d+\s+ms\)\s*$', line, re.IGNORECASE):
                info_dict['online'] += 1
            else:
                mobj = re.search('(unreachable|lagged|unknown|unmonitored)\s*$', 
                                 line, re.IGNORECASE)
                if mobj:
                    info_dict[mobj.group(1).lower()] += 1
                
        return info_dict