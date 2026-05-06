def trade_history(
        self, from_=None, count=None, from_id=None, end_id=None,
        order=None, since=None, end=None, pair=None
    ):
        """
        Returns trade history.
        To use this method you need a privilege of the info key.

        :param int or None from_: trade ID, from which the display starts (default 0)
        :param int or None count: the number of trades for display	(default 1000)
        :param int or None from_id: trade ID, from which the display starts	(default 0)
        :param int or None end_id: trade ID on which the display ends (default inf.)
        :param str or None order: sorting (default 'DESC')
        :param int or None since: the time to start the display (default 0)
        :param int or None end: the time to end the display	(default inf.)
        :param str or None pair: pair to be displayed (ex. 'btc_usd')
        """
        return self._trade_api_call(
            'TradeHistory', from_=from_, count=count, from_id=from_id, end_id=end_id,
            order=order, since=since, end=end, pair=pair
        )