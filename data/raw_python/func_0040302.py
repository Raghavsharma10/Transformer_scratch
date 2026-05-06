def load(self, page = None, verbose=False):
        """
        call to execute the collection loading
        :param page: integer of the page to load
        :param verbose: boolean to print to console
        :returns response
        :raises the SalesKingException
        """
        url = self._build_query_url(page, verbose)
        response = self._load(url, verbose)
        response = self._post_load(response, verbose)
        return response