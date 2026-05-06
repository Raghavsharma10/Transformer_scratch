def create_account(self, currency=None):
        """ Create a new account.

            This call is only available on the sandbox system. Please
            create accounts on fxtrade.oanda.com on our production
            system.

            See more:
            http://developer.oanda.com/rest-sandbox/accounts/#-a-name-createtestaccount-a-create-a-test-account
        """
        url = "{0}/{1}/accounts".format(self.domain, self.API_VERSION)
        params = {"currency": currency}
        try:
            return self._Client__call(uri=url, params=params, method="post")
        except RequestException:
            return False
        except AssertionError:
            return False