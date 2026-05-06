def get_trades(self, max_id=None, count=None, instrument=None, ids=None):
        """ Get a list of open trades

            Parameters
            ----------
            max_id : int
                The server will return trades with id less than or equal
                to this, in descending order (for pagination)
            count : int
                Maximum number of open trades to return. Default: 50 Max
                value: 500
            instrument : str
                Retrieve open trades for a specific instrument only
                Default: all
            ids : list
                A list of trades to retrieve. Maximum number of ids: 50.
                No other parameter may be specified with the ids
                parameter.

            See more:
            http://developer.oanda.com/rest-live/trades/#getListOpenTrades
        """
        url = "{0}/{1}/accounts/{2}/trades".format(
            self.domain,
            self.API_VERSION,
            self.account_id
        )
        params = {
            "maxId": int(max_id) if max_id and max_id > 0 else None,
            "count": int(count) if count and count > 0 else None,
            "instrument": instrument,
            "ids": ','.join(ids) if ids else None
        }

        try:
            return self._Client__call(uri=url, params=params, method="get")
        except RequestException:
            return False
        except AssertionError:
            return False