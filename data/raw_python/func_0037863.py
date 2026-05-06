def deleteAll(self):
        """
        Deletes whole Solr index. Use with care.
        """
        for core in self.endpoints:
            self._send_solr_command(self.endpoints[core], "{\"delete\": { \"query\" : \"*:*\"}}")