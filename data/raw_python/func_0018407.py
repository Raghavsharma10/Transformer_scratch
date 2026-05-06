def create_order(self, order):
        """
            See more:
            http://developer.oanda.com/rest-live/orders/#createNewOrder
        """
        url = "{0}/{1}/accounts/{2}/orders".format(
            self.domain,
            self.API_VERSION,
            self.account_id
        )
        try:
            return self._Client__call(
                uri=url,
                params=order.__dict__,
                method="post"
            )
        except RequestException:
            return False
        except AssertionError:
            return False