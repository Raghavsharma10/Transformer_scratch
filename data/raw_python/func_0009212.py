def add_batch(self, loans, batch_amount=None):
        """
        Add a batch of loans to your order.

        Parameters
        ----------
        loans : list
            A list of dictionary objects representing each loan and the amount you want to invest in it (see examples below).
        batch_amount : int, optional
            The dollar amount you want to set on ALL loans in this batch.
            **NOTE:** This will override the invest_amount value for each loan.

        Examples
        --------
        Each item in the loans list can either be a loan ID OR a dictionary object containing `loan_id` and
        `invest_amount` values. The invest_amount value is the dollar amount you wish to invest in this loan.

        **List of IDs**::

            # Invest $50 in 3 loans
            order.add_batch([1234, 2345, 3456], 50)

        **List of Dictionaries**::

            # Invest different amounts in each loans
            order.add_batch([
                {'loan_id': 1234, invest_amount: 50},
                {'loan_id': 2345, invest_amount: 25},
                {'loan_id': 3456, invest_amount: 150}
            ])
        """
        assert batch_amount is None or batch_amount % 25 == 0, 'batch_amount must be a multiple of 25'

        # Add each loan
        assert type(loans) is list, 'The loans property must be a list. (not {0})'.format(type(loans))
        for loan in loans:
            loan_id = loan
            amount = batch_amount

            # Extract ID and amount from loan dict
            if type(loan) is dict:
                assert 'loan_id' in loan, 'Each loan dict must have a loan_id value'
                assert batch_amount or 'invest_amount' in loan, 'Could not determine how much to invest in loan {0}'.format(loan['loan_id'])

                loan_id = loan['loan_id']
                if amount is None and 'invest_amount' in loan:
                    amount = loan['invest_amount']

            assert amount is not None, 'Could not determine how much to invest in loan {0}'.format(loan_id)
            assert amount % 25 == 0, 'Amount to invest must be a multiple of 25 (loan_id: {0})'.format(loan_id)

            self.add(loan_id, amount)