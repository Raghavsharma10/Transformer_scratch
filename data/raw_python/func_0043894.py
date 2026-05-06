def depth(self, pair, limit=150, ignore_invalid=0):
        """
        This method provides the information about active orders on the pair.
        :param str or iterable pair: pair (ex. 'btc_usd' or ['btc_usd', 'eth_usd'])
        :param limit: how many orders should be displayed (150 by default, max 5000)
        :param int ignore_invalid: ignore non-existing pairs
        """
        return self._public_api_call('depth', pair=pair, limit=limit, ignore_invalid=ignore_invalid)