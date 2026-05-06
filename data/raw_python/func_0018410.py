def request_transaction_history(self):
        """ Request full account history.

            Submit a request for a full transaction history.  A
            successfully accepted submission results in a response
            containing a URL in the Location header to a file that will
            be available once the request is served. Response for the
            URL will be HTTP 404 until the file is ready. Once served
            the URL will be valid for a certain amount of time.

            See more:
            http://developer.oanda.com/rest-live/transaction-history/#getFullAccountHistory
            http://developer.oanda.com/rest-live/transaction-history/#transactionTypes
        """
        url = "{0}/{1}/accounts/{2}/alltransactions".format(
            self.domain,
            self.API_VERSION,
            self.account_id
        )
        try:
            resp = self.__get_response(url)
            return resp.headers['location']
        except RequestException:
            return False
        except AssertionError:
            return False