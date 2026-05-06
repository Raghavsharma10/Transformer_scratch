def retrieveVals(self):
        """Retrieve values for graphs."""
        if self._diskList:
            self._fetchDevAll('disk', self._diskList, 
                              self._info.getDiskStats)
        if self._mdList:
            self._fetchDevAll('md', self._mdList, 
                              self._info.getMDstats)
        if self._partList:
            self._fetchDevAll('part', self._partList, 
                              self._info.getPartitionStats) 
        if self._lvList:
            self._fetchDevAll('lv', self._lvList, 
                              self._info.getLVstats)
        self._fetchDevAll('fs', self._fsList, 
                          self._info.getFilesystemStats)