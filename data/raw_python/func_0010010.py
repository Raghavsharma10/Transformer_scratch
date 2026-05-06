def check_url(url):
        '''
        Check if resource at URL is fetchable. (by trying to fetch it and checking for 200 status.

        Args:
            url (str): Url to check.

        Returns:
            Returns a tuple of {True/False, response code}

        '''
        request = urllib2.Request(url)
        try:
            response = urlopen(request)
            return True, response.code
        except urllib2.HTTPError as e:
            return False, e.code