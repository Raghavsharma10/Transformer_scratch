def auto_complete(self, term, state=None, postcode=None, max_results=None):
        """
        Gets a list of addresses that begin with the given term.
        """
        self._validate_state(state)
        params = {"term": term, "state": state, "postcode": postcode,
                  "max_results": max_results or self.max_results}
        return self._make_request('/address/autoComplete', params)