def trans_history(
        self, from_=None, count=None, from_id=None, end_id=None,
        order=None, since=None, end=None
    ):
        """
        Returns the history of transactions.
        To use this method you need a privilege of the info key.

        :param int or None from_: transaction ID, from which the display starts (default 0)
        :param int or None count: number of transaction to be displayed	(default 1000)
        :param int or None from_id: transaction ID, from which the display starts (default 0)
        :param int or None end_id: transaction ID on which the display ends	(default inf.)
        :param str or None order: sorting (default 'DESC')
        :param int or None since: the time to start the display (default 0)
        :param int or None end: the time to end the display	(default inf.)
        """
        return self._trade_api_call(
            'TransHistory', from_=from_, count=count, from_id=from_id, end_id=end_id,
            order=order, since=since, end=end
        )