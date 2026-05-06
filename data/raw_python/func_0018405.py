def get_orders(self, instrument=None, count=50):
        """
            See more:
            http://developer.oanda.com/rest-live/orders/#getOrdersForAnAccount
        """
        url = "{0}/{1}/accounts/{2}/orders".format(
            self.domain,
            self.API_VERSION,
            self.account_id
        )
        params = {"instrument": instrument, "count": count}
        try:
            return self._Client__call(uri=url, params=params, method="get")
        except RequestException:
            return False
        except AssertionError:
            return False