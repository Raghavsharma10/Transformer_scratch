def get_accounts(self, username=None):
        """ Get a list of accounts owned by the user.

            Parameters
            ----------
            username : string
                The name of the user. Note: This is only required on the
                sandbox, on production systems your access token will
                identify you.

            See more:
            http://developer.oanda.com/rest-sandbox/accounts/#-a-name-getaccountsforuser-a-get-accounts-for-a-user
        """
        url = "{0}/{1}/accounts".format(self.domain, self.API_VERSION)
        params = {"username": username}
        try:
            return self._Client__call(uri=url, params=params, method="get")
        except RequestException:
            return False
        except AssertionError:
            return False