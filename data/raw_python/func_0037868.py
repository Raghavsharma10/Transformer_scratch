def query(self, query, filters=None, columns=None, sort=None, start=0, rows=30):
        """
        Queries Solr and returns results

        query - Text query to search for
        filters - dictionary of filters to apply when searching in form of { "field":"filter_value" }
        columns - columns to return, list of strings
        sort - list of fields to sort on in format of ["field asc", "field desc", ... ]
        start - start number of first result (used in pagination)
        rows - number of rows to return (used for pagination, defaults to 30)
        """

        if not columns:
            columns = ["*", "score"]

        fields = {"q": query,
                 "json.nl" :"map",           # Return facets as JSON objects
                 "fl": ",".join(columns),    # Return score along with results
                 "start": str(start),
                 "rows": str(rows),
                 "wt": "json"}

        # Use shards parameter only if there are several cores active
        if len(self.endpoints) > 1:
            fields["shards"] = self._get_shards()

        # Prepare filters
        if not filters is None:
            filter_list = []
            for filter_field, value in filters.items():
                filter_list.append("%s:%s" % (filter_field, value))
            fields["fq"] = " AND ".join(filter_list)

        # Append sorting parameters
        if not sort is None:
            fields["sort"] = ",".join(sort)

        # Do request to Solr server to default endpoint (other cores will be queried with shard functionality)
        assert self.default_endpoint in self.endpoints
        request_url = _get_url(self.endpoints[self.default_endpoint], "select")
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