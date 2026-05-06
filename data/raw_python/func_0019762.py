def getCPUuse(self):
        """Return cpu time utilization in seconds.
        
        @return: Dictionary of stats.
        
        """
        hz = os.sysconf('SC_CLK_TCK')
        info_dict = {}
        try:
            fp = open(cpustatFile, 'r')
            line = fp.readline()
            fp.close()
        except:
            raise IOError('Failed reading stats from file: %s' % cpustatFile)
        headers = ['user', 'nice', 'system', 'idle', 'iowait', 'irq', 'softirq', 'steal', 'guest']
        arr = line.split()
        if len(arr) > 1 and arr[0] == 'cpu':
            return dict(zip(headers[0:len(arr)], [(float(t) / hz) for t in arr[1:]]))
        return info_dict