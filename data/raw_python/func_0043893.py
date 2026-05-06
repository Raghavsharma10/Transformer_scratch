def ticker(self, pair, ignore_invalid=0):
        """
        This method provides all the information about currently active pairs, such as: the maximum price,
        the minimum price, average price, trade volume, trade volume in currency, the last trade, Buy and Sell price.
        All information is provided over the past 24 hours.
        :param str or iterable pair: pair (ex. 'btc_usd' or ['btc_usd', 'eth_usd'])
        :param int ignore_invalid: ignore non-existing pairs
        """
        return self._public_api_call('ticker', pair=pair, ignore_invalid=ignore_invalid)