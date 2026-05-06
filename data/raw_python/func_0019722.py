def getSwapStats(self, dev):
        """Returns I/O stats for swap partition.
        
        @param dev: Device name for swap partition.
        @return: Dict of stats.
        
        """
        if self._swapList is None:
            self._initSwapInfo()
        if dev in self._swapList:
            return self.getDevStats(dev)
        else:
            return None