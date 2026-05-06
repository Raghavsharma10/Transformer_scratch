def get_order(self, order_id):
        """
            See more:
            http://developer.oanda.com/rest-live/orders/#getInformationForAnOrder
        """
        url = "{0}/{1}/accounts/{2}/orders/{3}".format(
            self.domain,
            self.API_VERSION,
            self.account_id,
            order_id
        )
        try:
            return self._Client__call(uri=url, method="get")
        except RequestException:
            return False
        except AssertionError:
            return False