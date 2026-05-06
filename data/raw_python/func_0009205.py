def assign_to_portfolio(self, portfolio_name, loan_id, order_id):
        """
        Assign a note to a named portfolio. `loan_id` and `order_id` can be either
        integer values or lists. If choosing lists, they both **MUST** be the same length
        and line up. For example, `order_id[5]` must be the order ID for `loan_id[5]`

        Parameters
        ----------
        portfolio_name : string
            The name of the portfolio to assign a the loan note to -- new or existing
        loan_id : int or list
            The loan ID, or list of loan IDs, to assign to the portfolio
        order_id : int or list
            The order ID, or list of order IDs, that this loan note was invested with.
            You can find this in the dict returned from `get_note()`

        Returns
        -------
        boolean
            True on success
        """
        response = None

        assert type(loan_id) == type(order_id), "Both loan_id and order_id need to be the same type"
        assert type(loan_id) in (int, list), "loan_id and order_id can only be int or list types"
        assert type(loan_id) is int or (type(loan_id) is list and len(loan_id) == len(order_id)), "If order_id and loan_id are lists, they both need to be the same length"

        # Data
        post = {
            'loan_id': loan_id,
            'record_id': loan_id,
            'order_id': order_id
        }
        query = {
            'method': 'createLCPortfolio',
            'lcportfolio_name': portfolio_name
        }

        # Is it an existing portfolio
        existing = self.get_portfolio_list()
        for folio in existing:
            if folio['portfolioName'] == portfolio_name:
                query['method'] = 'addToLCPortfolio'

        # Send
        response = self.session.post('/data/portfolioManagement', query=query, data=post)
        json_response = response.json()

        # Failed
        if not self.session.json_success(json_response):
            raise LendingClubError('Could not assign order to portfolio \'{0}\''.format(portfolio_name), response)

        # Success
        else:

            # Assigned to another portfolio, for some reason, raise warning
            if 'portfolioName' in json_response and json_response['portfolioName'] != portfolio_name:
                raise LendingClubError('Added order to portfolio "{0}" - NOT - "{1}", and I don\'t know why'.format(json_response['portfolioName'], portfolio_name))

            # Assigned to the correct portfolio
            else:
                self.__log('Added order to portfolio "{0}"'.format(portfolio_name))

            return True

        return False