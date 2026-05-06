def _do_search(self):
        """
        Perform the search, then convert that raw format into a
        SearchResults instance and return it.
        """
        if self._results_cache is None:
            response = self.raw()
            ResultsClass = self.get_results_class()
            results = self.to_python(response.get('hits', {}).get('hits', []))
            self._results_cache = ResultsClass(
                self.type, response, results, self.fields)
        return self._results_cache