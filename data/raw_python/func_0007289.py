def balance(self, account_id=None):
        """
        Returns balance information for a specific account.

        Official docs:
            https://monzo.com/docs/#read-balance

        :param account_id: Monzo account ID
        :type account_id: str
        :raises: ValueError
        :returns: Monzo balance instance
        :rtype: MonzoBalance
        """
        if not account_id:
            if len(self.accounts()) == 1:
                account_id = self.accounts()[0].id
            else:
                raise ValueError("You need to pass account ID")

        endpoint = '/balance'
        response = self._get_response(
            method='get', endpoint=endpoint,
            params={
                'account_id': account_id,
            },
        )

        return MonzoBalance(data=response.json())