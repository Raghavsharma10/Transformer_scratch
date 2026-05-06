def getIfaceStats(self):
        """Return dictionary of Traffic Stats for each Wanpipe Interface.
        
        @return: Nested dictionary of statistics for each interface.
        
        """
        ifInfo = netiface.NetIfaceInfo()
        ifStats = ifInfo.getIfStats()
        info_dict = {}
        for ifname in ifStats:
            if re.match('^w\d+g\d+$', ifname):
                info_dict[ifname] = ifStats[ifname]        
        return info_dict