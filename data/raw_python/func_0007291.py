def transactions(self, account_id=None, reverse=True, limit=None):
        """
        Returns a list of transactions on the user's account.

        Official docs:
            https://monzo.com/docs/#list-transactions

        :param account_id: Monzo account ID
        :type account_id: str
        :param reverse: whether transactions should be in in descending order
        :type reverse: bool
        :param limit: how many transactions should be returned; None for all
        :type limit: int
        :returns: list of Monzo transactions
        :rtype: list of MonzoTransaction
        """
        if not account_id:
            if len(self.accounts()) == 1:
                account_id = self.accounts()[0].id
            else:
                raise ValueError("You need to pass account ID")

        endpoint = '/transactions'
        response = self._get_response(
            method='get', endpoint=endpoint,
            params={
                'account_id': account_id,
            },
        )

        # The API does not allow reversing the list or limiting it, so to do
        # the basic query of 'get the latest transaction' we need to always get
        # all transactions and do the reversing and slicing in Python
        # I send Monzo an email, we'll se how they'll respond
        transactions = response.json()['transactions']
        if reverse:
            transactions.reverse()

        if limit:
            transactions = transactions[:limit]

        return [MonzoTransaction(data=t) for t in transactions]