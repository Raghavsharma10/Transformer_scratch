def getProcessStats(self):
        """Return stats for running and blocked processes, forks, 
        context switches and interrupts.
        
        @return: Dictionary of stats.
        
        """
        info_dict = {}
        try:
            fp = open(cpustatFile, 'r')
            data = fp.read()
            fp.close()
        except:
            raise IOError('Failed reading stats from file: %s' % cpustatFile)
        for line in data.splitlines():
            arr = line.split()
            if len(arr) > 1 and arr[0] in ('ctxt', 'intr', 'softirq',
                                           'processes', 'procs_running', 
                                           'procs_blocked'):
                info_dict[arr[0]] = arr[1]
        return info_dict