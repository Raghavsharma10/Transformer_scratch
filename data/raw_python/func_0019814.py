def retrieveVals(self):
        """Retrieve values for graphs."""
        file_stats = self._fileInfo.getContainerStats()
        for contname in self._fileContList:
            stats = file_stats.get(contname)
            if stats is not None:
                if self.hasGraph('rackspace_cloudfiles_container_size'):
                    self.setGraphVal('rackspace_cloudfiles_container_size', contname,
                                     stats.get('size'))
                if self.hasGraph('rackspace_cloudfiles_container_count'):
                    self.setGraphVal('rackspace_cloudfiles_container_count', contname,
                                     stats.get('count'))