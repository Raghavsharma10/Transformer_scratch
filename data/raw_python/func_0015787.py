def _do_search(self):
        """
        Perform the mlt call, then convert that raw format into a
        SearchResults instance and return it.
        """
        if self._results_cache is None:
            response = self.raw()
            results = self.to_python(response.get('hits', {}).get('hits', []))
            self._results_cache = DictSearchResults(
                self.type, response, results, None)
        return self._results_cache