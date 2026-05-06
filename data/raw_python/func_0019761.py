def getLoadAvg(self):
        """Return system Load Average.
        
        @return: List of 1 min, 5 min and 15 min Load Average figures.
        
        """
        try:
            fp = open(loadavgFile, 'r')
            line = fp.readline()
            fp.close()
        except:
            raise IOError('Failed reading stats from file: %s' % loadavgFile)
        arr = line.split()
        if len(arr) >= 3:
            return [float(col) for col in arr[:3]]
        else:
            return None