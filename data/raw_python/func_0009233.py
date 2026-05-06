def all_filters(lc):
        """
        Get a list of all your saved filters

        Parameters
        ----------
        lc : :py:class:`lendingclub.LendingClub`
            An instance of the authenticated LendingClub class

        Returns
        -------
        list
            A list of lendingclub.filters.SavedFilter objects
        """

        filters = []
        response = lc.session.get('/browse/getSavedFiltersAj.action')
        json_response = response.json()

        # Load all filters
        if lc.session.json_success(json_response):
            for saved in json_response['filters']:
                filters.append(SavedFilter(lc, saved['id']))

        return filters