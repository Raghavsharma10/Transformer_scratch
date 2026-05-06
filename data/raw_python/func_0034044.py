def similar(self, address_line, max_results=None):
        """
        Gets a list of valid addresses that are similar to the given term, can
        be used to match invalid addresses to valid addresses.
        """
        params = {"term": address_line,
                  "max_results": max_results or self.max_results}
        return self._make_request('/address/getSimilar', params)