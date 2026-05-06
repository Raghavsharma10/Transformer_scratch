def getVMstats(self):
        """Return stats for Virtual Memory Subsystem.
        
        @return: Dictionary of stats.
        
        """
        info_dict = {}
        try:
            fp = open(vmstatFile, 'r')
            data = fp.read()
            fp.close()
        except:
            raise IOError('Failed reading stats from file: %s' % vmstatFile)
        for line in data.splitlines():
            cols = line.split()
            if len(cols) == 2:
                info_dict[cols[0]] = cols[1]
        return info_dict