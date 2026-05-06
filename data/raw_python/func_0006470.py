def remove(self, document_id, namespace, timestamp):
        """Removes documents from Solr

        The input is a python dictionary that represents a mongo document.
        """
        self.solr.delete(id=u(document_id),
                         commit=(self.auto_commit_interval == 0))