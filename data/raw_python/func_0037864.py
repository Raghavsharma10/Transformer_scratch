def delete(self, id):
        """
        Deletes document with ID on all Solr cores
        """
        for core in self.endpoints:
            self._send_solr_command(self.endpoints[core], "{\"delete\" : { \"id\" : \"%s\"}}" % (id,))