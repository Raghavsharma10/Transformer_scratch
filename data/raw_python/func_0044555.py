def canonical_query_string(self):
        """
        The canonical query string from the query parameters.

        This takes the query string from the request and orders the parameters
        in 
        """
        results = []
        for key, values in iteritems(self.query_parameters):
            # Don't include the signature itself.
            if key == _x_amz_signature:
                continue
            
            for value in values:
                results.append("%s=%s" % (key, value))

        return "&".join(sorted(results))