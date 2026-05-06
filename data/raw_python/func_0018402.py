def get_instruments(self):
        """
            See more:
            http://developer.oanda.com/rest-live/rates/#getInstrumentList
        """
        url = "{0}/{1}/instruments".format(self.domain, self.API_VERSION)
        params = {"accountId": self.account_id}
        try:
            response = self._Client__call(uri=url, params=params)
            assert len(response) > 0
            return response
        except RequestException:
            return False
        except AssertionError:
            return False