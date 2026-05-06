def _stream_search(self, query):
        """Helper method for iterating over Solr search results."""
        for doc in self.solr.search(query, rows=100000000):
            if self.unique_key != "_id":
                doc["_id"] = doc.pop(self.unique_key)
            yield doc