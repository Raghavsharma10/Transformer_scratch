def getSwapStats(self):
        """Return information on swap partition and / or files.
        
            @return: Dictionary of stats.
            
        """
        info_dict = {}
        try:
            fp = open(swapsFile, 'r')
            data = fp.read()
            fp.close()
        except:
            raise IOError('Failed reading stats from file: %s' % swapsFile)
        lines = data.splitlines()
        if len(lines) > 1:
            colnames = [name.lower() for name in lines[0].split()]
            for line in lines[1:]:
                cols = line.split()
                info_dict[cols[0]] = dict(zip(colnames[1:], cols[1:]))
        return info_dict