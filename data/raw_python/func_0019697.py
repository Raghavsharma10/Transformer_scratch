def retrieveVals(self):
        """Retrieve values for graphs."""
        name = 'diskspace'
        if self.hasGraph(name):
            for fspath in self._fslist:
                if self._statsSpace.has_key(fspath):
                    self.setGraphVal(name, fspath, 
                                     self._statsSpace[fspath]['inuse_pcent'])
        name = 'diskinode'
        if self.hasGraph(name):
            for fspath in self._fslist:
                if self._statsInode.has_key(fspath):
                    self.setGraphVal(name, fspath, 
                                     self._statsInode[fspath]['inuse_pcent'])