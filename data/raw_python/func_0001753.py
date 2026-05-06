def send_request(self, endpoint='ticker', coin_name=None, **kwargs):
        """: param string 'ticker', it's 'ticker' if we want info about coins,
            'global' for global market's info.
           : param string 'coin_name', specify the name of the coin, if None,
             we'll retrieve info about all available coins.
        """

        built_url = self._make_url(endpoint, coin_name)
        payload = dict(**kwargs)

        self._process_request(endpoint, built_url, payload)