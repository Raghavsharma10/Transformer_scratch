def my_notes(self, start_index=0, limit=100, get_all=False, sort_by='loanId', sort_dir='asc'):
        """
        Return all the loan notes you've already invested in. By default it'll return 100 results at a time.

        Parameters
        ----------
        start_index : int, optional
            The result index to start on. By default only 100 records will be returned at a time, so use this
            to start at a later index in the results. For example, to get results 200 - 300, set `start_index` to 200.
            (default is 0)
        limit : int, optional
            The number of results to return per request. (default is 100)
        get_all : boolean, optional
            Return all results in one request, instead of 100 per request.
        sort_by : string, optional
            What key to sort on
        sort_dir : {'asc', 'desc'}, optional
            Which direction to sort

        Returns
        -------
        dict
            A dictionary with a list of matching notes on the `loans` key
        """

        index = start_index
        notes = {
            'loans': [],
            'total': 0,
            'result': 'success'
        }
        while True:
            payload = {
                'sortBy': sort_by,
                'dir': sort_dir,
                'startindex': index,
                'pagesize': limit,
                'namespace': '/account'
            }
            response = self.session.post('/account/loansAj.action', data=payload)
            json_response = response.json()

            # Notes returned
            if self.session.json_success(json_response):
                notes['loans'] += json_response['searchresult']['loans']
                notes['total'] = json_response['searchresult']['totalRecords']

            # Error
            else:
                notes['result'] = json_response['result']
                break

            # Load more
            if get_all is True and len(notes['loans']) < notes['total']:
                index += limit

            # End
            else:
                break

        return notes