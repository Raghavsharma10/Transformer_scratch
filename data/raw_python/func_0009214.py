def assign_to_portfolio(self, portfolio_name=None):
        """
        Assign all the notes in this order to a portfolio

        Parameters
        ----------
            portfolio_name -- The name of the portfolio to assign it to (new or existing)

        Raises
        ------
        LendingClubError

        Returns
        -------
        boolean
            True on success
        """
        assert self.order_id > 0, 'You need to execute this order before you can assign to a portfolio.'

        # Get loan IDs as a list
        loan_ids = self.loans.keys()

        # Make a list of 1 order ID per loan
        order_ids = [self.order_id]*len(loan_ids)

        return self.lc.assign_to_portfolio(portfolio_name, loan_ids, order_ids)