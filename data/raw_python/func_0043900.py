def create_coupon(self, currency, amount, receiver):
        """
        This method allows you to create Coupons.
        Please, note: In order to use this method, you need the Coupon key privilege. You can make a request to
        enable it by submitting a ticket to Support..
        You need to create the API key that you are going to use for this method in advance. Please provide
        the first 8 characters of the key (e.g. HKG82W66) in your ticket to support. We'll enable the Coupon privilege
        for this key.
        You must also provide us the IP-addresses from which you will be accessing the API.
        When using this method, there will be no additional confirmations of transactions. Please note that you are
        fully responsible for keeping the secret of the API key safe after we have enabled the Withdraw
        privilege for it.

        :param str currency: currency (ex. 'BTC')
        :param int amount: withdrawal amount
        :param str receiver: name of user who is allowed to redeem the code
        """
        return self._trade_api_call('CreateCoupon', currency=currency, amount=amount, receiver=receiver)