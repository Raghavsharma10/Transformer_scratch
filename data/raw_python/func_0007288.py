def accounts(self, refresh=False):
        """
        Returns a list of accounts owned by the currently authorised user.
        It's often used when deciding whether to require explicit account ID
        or use the only available one, so we cache the response by default.

        Official docs:
            https://monzo.com/docs/#list-accounts

        :param refresh: decides if the accounts information should be refreshed
        :type refresh: bool
        :returns: list of Monzo accounts
        :rtype: list of MonzoAccount
        """
        if not refresh and self._cached_accounts:
            return self._cached_accounts

        endpoint = '/accounts'
        response = self._get_response(
            method='get', endpoint=endpoint,
        )

        accounts_json = response.json()['accounts']
        accounts = [MonzoAccount(data=account) for account in accounts_json]
        self._cached_accounts = accounts

        return accounts