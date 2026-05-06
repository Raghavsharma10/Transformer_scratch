def more_like_this(self, query, fields, columns=None, start=0, rows=30):
        """
        Retrieves "more like this" results for a passed query document

        query - query for a document on which to base similar documents
        fields - fields on which to base similarity estimation (either comma delimited string or a list)
        columns - columns to return (list of strings)
        start - start number for first result (used in pagination)
        rows - number of rows to return (used for pagination, defaults to 30)
        """
        if isinstance(fields, basestring):
            mlt_fields = fields
        else:
            mlt_fields = ",".join(fields)

        if columns is None:
            columns = ["*", "score"]

        fields = {'q' : query,
                  'json.nl': 'map',
                  'mlt.fl': mlt_fields,
                  'fl': ",".join(columns),
                  'start': str(start),
                  'rows': str(rows),
                  'wt': "json"}

        if len(self.endpoints) > 1:
            fields["shards"] = self._get_shards()

        assert self.default_endpoint in self.endpoints
        request_url = _get_url(self.endpoints[self.default_endpoint], "mlt")
        results = self._send_solr_query(request_url, fields)
        if not results:
            return None

        assert "responseHeader" in results
        # Check for response status
        if not results.get("responseHeader").get("status") == 0:
            logger.error("Server error while retrieving results: %s", results)
            return None

        assert "response" in results

        result_obj = self._parse_response(results)
        return result_obj