def update_trade(
        self,
        trade_id,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None
    ):
        """ Modify an existing trade.

            Note: Only the specified parameters will be modified. All
            other parameters will remain unchanged. To remove an
            optional parameter, set its value to 0.

            Parameters
            ----------
            trade_id : int
                The id of the trade to modify.
            stop_loss : number
                Stop Loss value.
            take_profit : number
                Take Profit value.
            trailing_stop : number
                Trailing Stop distance in pips, up to one decimal place

            See more:
            http://developer.oanda.com/rest-live/trades/#modifyExistingTrade
        """
        url = "{0}/{1}/accounts/{2}/trades/{3}".format(
            self.domain,
            self.API_VERSION,
            self.account_id,
            trade_id
        )
        params = {
            "stopLoss": stop_loss,
            "takeProfit": take_profit,
            "trailingStop": trailing_stop
        }
        try:
            return self._Client__call(uri=url, params=params, method="patch")
        except RequestException:
            return False
        except AssertionError:
            return False
        raise NotImplementedError()