def search(self, keyword, types=[], terr=KKBOXTerritory.TAIWAN):
        '''
        Searches within KKBOX's database.

        :param keyword: the keyword.
        :type keyword: str
        :param types: the search types.
        :return: list
        :param terr: the current territory.
        :return: API response.
        :rtype: dict

        See `https://docs-en.kkbox.codes/v1.1/reference#search_1`.
        '''
        url = 'https://api.kkbox.com/v1.1/search'
        url += '?' + url_parse.urlencode({'q': keyword, 'territory': terr})
        if len(types) > 0:
            url += '&type=' + ','.join(types)
        return self.http._post_data(url, None, self.http._headers_with_access_token())