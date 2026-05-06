def getIfStats(self):
        """Return dictionary of Traffic Stats for Network Interfaces.
        
        @return: Nested dictionary of statistics for each interface.
        
        """
        info_dict = {}
        try:
            fp = open(ifaceStatsFile, 'r')
            data = fp.read()
            fp.close()
        except:
            raise IOError('Failed reading interface stats from file: %s'
                          % ifaceStatsFile)
        for line in data.splitlines():
            mobj = re.match('^\s*([\w\d:]+):\s*(.*\S)\s*$', line)
            if mobj:
                iface = mobj.group(1)
                statline = mobj.group(2)
                info_dict[iface] = dict(zip(
                    ('rxbytes', 'rxpackets', 'rxerrs', 'rxdrop', 'rxfifo',
                     'rxframe', 'rxcompressed', 'rxmulticast',
                     'txbytes', 'txpackets', 'txerrs', 'txdrop', 'txfifo',
                     'txcolls', 'txcarrier', 'txcompressed'),
                    [int(x) for x in statline.split()]))
                    
        return info_dict