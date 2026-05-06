def withdraw_coin(self, coin_name, amount, address):
        """
        The method is designed for cryptocurrency withdrawals.
        Please note: You need to have the privilege of the Withdraw key to be able to use this method. You can make
        a request for enabling this privilege by submitting a ticket to Support.
        You need to create the API key that you are going to use for this method in advance. Please provide the first
        8 characters of the key (e.g. HKG82W66) in your ticket to support. We'll enable the Withdraw privilege for
        this key.
        When using this method, there will be no additional confirmations of withdrawal. Please note that you are
        fully responsible for keeping the secret of the API key safe after we have enabled the Withdraw
        privilege for it.

        :param str coin_name: currency (ex. 'BTC')
        :param int amount: withdrawal amount
        :param str address: withdrawal address
        """
        return self._trade_api_call('WithdrawCoin', coinName=coin_name, amount=amount, address=address)