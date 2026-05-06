def get_request(self, url):
        """
        Wraps the URL to retrieve to protects against "creative"
        interpretation of the RFC: http://bugs.python.org/issue8732
        """
        if isinstance(url, string_types):
            url = urllib2.Request(url, headers={'Accept-encoding': 'identity'})
        return url