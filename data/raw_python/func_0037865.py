def commit(self):
        """
        Flushes all pending changes and commits Solr changes
        """
        self._addFlushBatch()
        for core in self.endpoints:
            self._send_solr_command(self.endpoints[core], "{ \"commit\":{} }")