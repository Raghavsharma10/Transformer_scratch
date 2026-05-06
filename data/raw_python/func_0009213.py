def execute(self, portfolio_name=None):
        """
        Place the order with LendingClub

        Parameters
        ----------
        portfolio_name : string
            The name of the portfolio to add the invested loan notes to.
            This can be a new or existing portfolio name.

        Raises
        ------
        LendingClubError

        Returns
        -------
        int
            The completed order ID
        """
        assert self.order_id == 0, 'This order has already been place. Start a new order.'
        assert len(self.loans) > 0, 'There aren\'t any loans in your order'

        # Place the order
        self.__stage_order()
        token = self.__get_strut_token()
        self.order_id = self.__place_order(token)

        self.__log('Order #{0} was successfully submitted'.format(self.order_id))

        # Assign to portfolio
        if portfolio_name:
            return self.assign_to_portfolio(portfolio_name)

        return self.order_id