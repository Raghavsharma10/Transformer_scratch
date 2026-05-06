def search(self, start_ts, end_ts):
        """Called to query Solr for documents in a time range."""
        query = '_ts: [%s TO %s]' % (start_ts, end_ts)
        return self._stream_search(query)