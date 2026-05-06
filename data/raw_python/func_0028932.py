def request(self, api_query, url=None):
        """
        e.g. {'action': 'query', 'meta': 'userinfo'}. format=json not required
        function returns a python dict that resembles the api's json response
        """
        api_query['format'] = 'json'
        if url is not None:
            api_url = url + "/api.php"
        else:
            api_url = self.api_url

        size = sum([sys.getsizeof(v) for k, v in iteritems(api_query)])

        if size > (1024 * 8):
            # if request is bigger than 8 kB (the limit is somewhat arbitrary,
            # see https://www.mediawiki.org/wiki/API:Edit#Large_texts) then
            # transmit as multipart message

            req = self._prepare_long_request(url=api_url,
                                             api_query=api_query)
            req.send()
            if self.return_json:
                return req.response.json()
            else:
                return req.response.text
        else:
            auth1 = OAuth1(
                self.consumer_token.key,
                client_secret=self.consumer_token.secret,
                resource_owner_key=session['mwoauth_access_token']['key'],
                resource_owner_secret=session['mwoauth_access_token']['secret'])
            if self.return_json:
                return requests.post(api_url, data=api_query, auth=auth1).json()
            else:
                return requests.post(api_url, data=api_query, auth=auth1).text