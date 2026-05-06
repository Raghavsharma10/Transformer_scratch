def get_prices(self, instruments, stream=True):
        """
            See more:
            http://developer.oanda.com/rest-live/rates/#getCurrentPrices
        """
        url = "{0}/{1}/prices".format(
            self.domain_stream if stream else self.domain,
            self.API_VERSION
        )
        params = {"accountId": self.account_id, "instruments": instruments}

        call = {"uri": url, "params": params, "method": "get"}

        try:
            if stream:
                return self._Client__call_stream(**call)
            else:
                return self._Client__call(**call)
        except RequestException:
            return False
        except AssertionError:
            return False