def upsert(self, doc, namespace, timestamp):
        """Update or insert a document into Solr

        This method should call whatever add/insert/update method exists for
        the backend engine and add the document in there. The input will
        always be one mongo document, represented as a Python dictionary.
        """
        if self.auto_commit_interval is not None:
            self.solr.add([self._clean_doc(doc, namespace, timestamp)],
                          commit=(self.auto_commit_interval == 0),
                          commitWithin=u(self.auto_commit_interval))
        else:
            self.solr.add([self._clean_doc(doc, namespace, timestamp)],
                          commit=False)