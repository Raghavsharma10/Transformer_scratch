def build_portfolio(self, cash, max_per_note=25, min_percent=0, max_percent=20, filters=None, automatically_invest=False, do_not_clear_staging=False):
        """
        Returns a list of loan notes that are diversified by your min/max percent request and filters.
        One way to invest in these loan notes, is to start an order and use add_batch to add all the
        loan fragments to them. (see examples)

        Parameters
        ----------
        cash : int
            The total amount you want to invest across a portfolio of loans (at least $25).
        max_per_note : int, optional
            The maximum dollar amount you want to invest per note. Must be a multiple of 25
        min_percent : int, optional
            THIS IS NOT PER NOTE, but the minimum average percent of return for the entire portfolio.
        max_percent : int, optional
            THIS IS NOT PER NOTE, but the maxmimum average percent of return for the entire portfolio.
        filters : lendingclub.filters.*, optional
            The filters to use to search for portfolios
        automatically_invest : boolean, optional
            If you want the tool to create an order and automatically invest in the portfolio that matches your filter.
            (default False)
        do_not_clear_staging : boolean, optional
            Similar to automatically_invest, don't do this unless you know what you're doing.
            Setting this to True stops the method from clearing the loan staging area before returning

        Returns
        -------
        dict
            A dict representing a new portfolio or False if nothing was found.
            If `automatically_invest` was set to `True`, the dict will contain an `order_id` key with
            the ID of the completed investment order.

        Notes
        -----
        **The min/max_percent parameters**

        When searching for portfolios, these parameters will match a portfolio of loan notes which have
        an **AVERAGE** percent return between these values. If there are multiple portfolio matches, the
        one closes to the max percent will be chosen.

        Examples
        --------
        Here we want to invest $400 in a portfolio with only B, C, D and E grade notes with an average overall return between 17% - 19%. This similar to finding a portfolio in the 'Invest' section on lendingclub.com::

            >>> from lendingclub import LendingClub
            >>> from lendingclub.filters import Filter
            >>> lc = LendingClub()
            >>> lc.authenticate()
            Email:test@test.com
            Password:
            True
            >>> filters = Filter()                  # Set the search filters (only B, C, D and E grade notes)
            >>> filters['grades']['C'] = True
            >>> filters['grades']['D'] = True
            >>> filters['grades']['E'] = True
            >>> lc.get_cash_balance()               # See the cash you have available for investing
            463.80000000000001

            >>> portfolio = lc.build_portfolio(400, # Invest $400 in a portfolio...
                    min_percent=17.0,               # Return percent average between 17 - 19%
                    max_percent=19.0,
                    max_per_note=50,                # As much as $50 per note
                    filters=filters)                # Search using your filters

            >>> len(portfolio['loan_fractions'])    # See how many loans are in this portfolio
            16
            >>> loans_notes = portfolio['loan_fractions']
            >>> order = lc.start_order()            # Start a new order
            >>> order.add_batch(loans_notes)        # Add the loan notes to the order
            >>> order.execute()                     # Execute the order
            1861880

        Here we do a similar search, but automatically invest the found portfolio. **NOTE** This does not allow
        you to review the portfolio before you invest in it.

            >>> from lendingclub import LendingClub
            >>> from lendingclub.filters import Filter
            >>> lc = LendingClub()
            >>> lc.authenticate()
            Email:test@test.com
            Password:
            True
                                                    # Filter shorthand
            >>> filters = Filter({'grades': {'B': True, 'C': True, 'D': True, 'E': True}})
            >>> lc.get_cash_balance()               # See the cash you have available for investing
            463.80000000000001

            >>> portfolio = lc.build_portfolio(400,
                    min_percent=17.0,
                    max_percent=19.0,
                    max_per_note=50,
                    filters=filters,
                    automatically_invest=True)      # Same settings, except invest immediately

            >>> portfolio['order_id']               # See order ID
            1861880
        """
        assert filters is None or isinstance(filters, Filter), 'filter is not a lendingclub.filters.Filter'
        assert max_per_note >= 25, 'max_per_note must be greater than or equal to 25'

        # Set filters
        if filters:
            filter_str = filters.search_string()
        else:
            filter_str = 'default'

        # Start a new order
        self.session.clear_session_order()

        # Make request
        payload = {
            'amount': cash,
            'max_per_note': max_per_note,
            'filter': filter_str
        }
        self.__log('POST VALUES -- amount: {0}, max_per_note: {1}, filter: ...'.format(cash, max_per_note))
        response = self.session.post('/portfolio/lendingMatchOptionsV2.action', data=payload)
        json_response = response.json()

        # Options were found
        if self.session.json_success(json_response) and 'lmOptions' in json_response:
            options = json_response['lmOptions']

            # Nothing found
            if type(options) is not list or json_response['numberTicks'] == 0:
                self.__log('No lending portfolios were returned with your search')
                return False

            # Choose an investment option based on the user's min/max values
            i = 0
            match_index = -1
            match_option = None
            for option in options:

                # A perfect match
                if option['percentage'] == max_percent:
                    match_option = option
                    match_index = i
                    break

                # Over the max
                elif option['percentage'] > max_percent:
                    break

                # Higher than the minimum percent and the current matched option
                elif option['percentage'] >= min_percent and (match_option is None or match_option['percentage'] < option['percentage']):
                    match_option = option
                    match_index = i

                i += 1

            # Nothing matched
            if match_option is None:
                self.__log('No portfolios matched your percentage requirements')
                return False

            # Mark this portfolio for investing (in order to get a list of all notes)
            payload = {
                'order_amount': cash,
                'lending_match_point': match_index,
                'lending_match_version': 'v2'
            }
            self.session.get('/portfolio/recommendPortfolio.action', query=payload)

            # Get all loan fractions
            payload = {
                'method': 'getPortfolio'
            }
            response = self.session.get('/data/portfolio', query=payload)
            json_response = response.json()

            # Extract fractions from response
            fractions = []
            if 'loanFractions' in json_response:
                fractions = json_response['loanFractions']

                # Normalize by converting loanFractionAmount to invest_amount
                for frac in fractions:
                    frac['invest_amount'] = frac['loanFractionAmount']

                    # Raise error if amount is greater than max_per_note
                    if frac['invest_amount'] > max_per_note:
                        raise LendingClubError('ERROR: LendingClub tried to invest ${0} in a loan note. Your max per note is set to ${1}. Portfolio investment canceled.'.format(frac['invest_amount'], max_per_note))

            if len(fractions) == 0:
                self.__log('The selected portfolio didn\'t have any loans')
                return False
            match_option['loan_fractions'] = fractions

            # Validate that fractions do indeed match the filters
            if filters is not None:
                filters.validate(fractions)

            # Not investing -- reset portfolio search session and return
            if automatically_invest is not True:
                if do_not_clear_staging is not True:
                    self.session.clear_session_order()

            # Invest in this porfolio
            elif automatically_invest is True:  # just to be sure
                order = self.start_order()

                # This should probably only be ever done here...ever.
                order._Order__already_staged = True
                order._Order__i_know_what_im_doing = True

                order.add_batch(match_option['loan_fractions'])
                order_id = order.execute()
                match_option['order_id'] = order_id

            return match_option
        else:
            raise LendingClubError('Could not find any portfolio options that match your filters', response)

        return False