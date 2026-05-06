def getSpaceUse(self):
        """Get disk space usage.
        
        @return: Dictionary of filesystem space utilization stats for filesystems.
        
        """
        stats = {}
        try:
            out = subprocess.Popen([dfCmd, "-Pk"], 
                                   stdout=subprocess.PIPE).communicate()[0]
        except:
            raise Exception('Execution of command %s failed.' % dfCmd)
        lines = out.splitlines()
        if len(lines) > 1:
            for line in lines[1:]:
                fsstats = {}
                cols = line.split()
                fsstats['device'] = cols[0]
                fsstats['type'] = self._fstypeDict[cols[5]]
                fsstats['total'] = 1024 * int(cols[1])
                fsstats['inuse'] = 1024 * int(cols[2])
                fsstats['avail'] = 1024 * int(cols[3])
                fsstats['inuse_pcent'] = int(cols[4][:-1])
                stats[cols[5]] = fsstats
        return stats