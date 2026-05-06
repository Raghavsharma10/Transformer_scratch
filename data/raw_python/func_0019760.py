def getUptime(self):
        """Return system uptime in seconds.
        
        @return: Float that represents uptime in seconds.
        
        """
        try:
            fp = open(uptimeFile, 'r')
            line = fp.readline()
            fp.close()
        except:
            raise IOError('Failed reading stats from file: %s' % uptimeFile)
        return float(line.split()[0])