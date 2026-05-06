def engineIncluded(self, name):
        """Utility method to check if a storage engine is included in graphs.
        
        @param name: Name of storage engine.
        @return:     Returns True if included in graphs, False otherwise.
            
        """
        if self._engines is None:
            self._engines = self._dbconn.getStorageEngines()
        return self.envCheckFilter('engine', name) and name in self._engines