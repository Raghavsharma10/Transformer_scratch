def fetch_next_page(self, data):
        '''
        Fetches next page based on previously fetched data.
        Will get the next page url from data['paging']['next'].

        :param data: previously fetched API response.
        :type data: dict        
        :return: API response.
        :rtype: dict
        '''
        next_url = data['paging']['next']
        if next_url != None:
            next_data = self.http._post_data(next_url, None, self.http._headers_with_access_token())
            return next_data
        else:
            return None