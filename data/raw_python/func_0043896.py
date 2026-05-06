def trade(self, pair, type_, rate, amount):
        """
        The basic method that can be used for creating orders and trading on the exchange.
        To use this method you need an API key privilege to trade.
        You can only create limit orders using this method, but you can emulate market orders using rate parameters.
        E.g. using rate=0.1 you can sell at the best market price.
        Each pair has a different limit on the minimum / maximum amounts, the minimum amount and the number of digits
        after the decimal point. All limitations can be obtained using the info method in PublicAPI v3.

        :param str pair: pair (ex. 'btc_usd')
        :param str type_: order type ('buy' or 'sell')
        :param float rate: the rate at which you need to buy/sell
        :param float amount: the amount you need to buy/sell
        """
        return self._trade_api_call('Trade', pair=pair, type_=type_, rate=rate, amount=amount)