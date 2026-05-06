def fetch_data(self, url):
        ''' 
        Fetches data from specific url.

        :return: The response.
        :rtype: dict
        '''
        return self.http._post_data(url, None, self.http._headers_with_access_token())