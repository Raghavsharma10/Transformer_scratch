def add(self, loan_id, amount):
        """
        Add a loan and amount you want to invest, to your order.
        If this loan is already in your order, it's amount will be replaced
        with the this new amount

        Parameters
        ----------
        loan_id : int or dict
            The ID of the loan you want to add or a dictionary containing a `loan_id` value
        amount : int % 25
            The dollar amount you want to invest in this loan, as a multiple of 25.
        """
        assert amount > 0 and amount % 25 == 0, 'Amount must be a multiple of 25'
        assert type(amount) in (float, int), 'Amount must be a number'

        if type(loan_id) is dict:
            loan = loan_id
            assert 'loan_id' in loan and type(loan['loan_id']) is int, 'loan_id must be a number or dictionary containing a loan_id value'
            loan_id = loan['loan_id']

        assert type(loan_id) in [str, unicode, int], 'Loan ID must be an integer number or a string'
        self.loans[loan_id] = amount