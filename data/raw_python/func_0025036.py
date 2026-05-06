def get_query_uri(self):
        """
        Return the uri used for queries on time series data.
        """
        # Query URI has extra path we don't want so strip it off here
        query_uri = self.service.settings.data['query']['uri']
        query_uri = urlparse(query_uri)
        return query_uri.scheme + '://' + query_uri.netloc