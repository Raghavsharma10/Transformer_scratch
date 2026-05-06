def getMemoryUse(self):
        """Return stats for memory utilization.
        
        @return: Dictionary of stats.
        
        """
        info_dict = {}
        try:
            fp = open(meminfoFile, 'r')
            data = fp.read()
            fp.close()
        except:
            raise IOError('Failed reading stats from file: %s' % meminfoFile)
        for line in data.splitlines():
            mobj = re.match('^(.+):\s*(\d+)\s*(\w+|)\s*$', line)
            if mobj:
                if mobj.group(3).lower() == 'kb':
                    mult = 1024
                else:
                    mult = 1
                info_dict[mobj.group(1)] = int(mobj.group(2)) * mult
        return info_dict