def getPartitionList(self):
        """Returns list of partitions.
        
        @return: List of (disk,partition) pairs.
        
        """
        if self._partList is None:
            self._partList = []
            for (disk,parts) in self.getPartitionDict().iteritems():
                for part in parts:
                    self._partList.append((disk,part))
        return self._partList