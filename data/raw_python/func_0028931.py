def _prepare_long_request(self, url, api_query):
        """
        Use requests.Request and requests.PreparedRequest to produce the
        body (and boundary value) of a multipart/form-data; POST request as
        detailed in https://www.mediawiki.org/wiki/API:Edit#Large_texts
        """

        partlist = []
        for k, v in iteritems(api_query):
            if k in ('title', 'text', 'summary'):
                # title,  text and summary values in the request
                # should be utf-8 encoded

                part = (k,
                        (None, v.encode('utf-8'),
                         'text/plain; charset=UTF-8',
                         {'Content-Transfer-Encoding': '8bit'}
                         )
                        )
            else:
                part = (k, (None, v))
            partlist.append(part)

        auth1 = OAuth1(
            self.consumer_token.key,
            client_secret=self.consumer_token.secret,
            resource_owner_key=session['mwoauth_access_token']['key'],
            resource_owner_secret=session['mwoauth_access_token']['secret'])
        return Request(
            url=url, files=partlist, auth=auth1, method="post").prepare()