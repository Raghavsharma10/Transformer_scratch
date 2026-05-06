def search(self, filters=None, start_index=0, limit=100):
        """
        Search for a list of notes that can be invested in.
        (similar to searching for notes in the Browse section on the site)

        Parameters
        ----------
        filters : lendingclub.filters.*, optional
            The filter to use to search for notes. If no filter is passed, a wildcard search
            will be performed.
        start_index : int, optional
            The result index to start on. By default only 100 records will be returned at a time, so use this
            to start at a later index in the results. For example, to get results 200 - 300, set `start_index` to 200.
            (default is 0)
        limit : int, optional
            The number of results to return per request. (default is 100)

        Returns
        -------
        dict
            A dictionary object with the list of matching loans under the `loans` key.
        """
        assert filters is None or isinstance(filters, Filter), 'filter is not a lendingclub.filters.Filter'

        # Set filters
        if filters:
            filter_string = filters.search_string()
        else:
            filter_string = 'default'
        payload = {
            'method': 'search',
            'filter': filter_string,
            'startindex': start_index,
            'pagesize': limit
        }

        # Make request
        response = self.session.post('/browse/browseNotesAj.action', data=payload)
        json_response = response.json()

        if self.session.json_success(json_response):
            results = json_response['searchresult']

            # Normalize results by converting loanGUID -> loan_id
            for loan in results['loans']:
                loan['loan_id'] = int(loan['loanGUID'])

            # Validate that fractions do indeed match the filters
            if filters is not None:
                filters.validate(results['loans'])

            return results

        return False