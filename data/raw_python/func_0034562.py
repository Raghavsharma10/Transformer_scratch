def request(self, path, data=None, method='GET'):
        """ Convenience Facebook request function. 

        Utility function to request resources via the graph API, with the
        format expected by Facebook.
        """
        url = '%s%s?access_token=%s' % (
            'https://graph.facebook.com',
            path,
            self['oauth_token'])

        req = Request(url, data=data)
        req.get_method = lambda: method

        return loads(urlopen(req).read())