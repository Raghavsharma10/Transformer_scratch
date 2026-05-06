def fetch_album(self, album_id, terr=KKBOXTerritory.TAIWAN):
        '''
        Fetches an album by given ID.

        :param album_id: the album ID.
        :type album_id: str
        :param terr: the current territory.
        :return: API response.
        :rtype: dict

        See `https://docs-en.kkbox.codes/v1.1/reference#albums-album_id`.
        '''
        url = 'https://api.kkbox.com/v1.1/albums/%s' % album_id
        url += '?' + url_parse.urlencode({'territory': terr})
        return self.http._post_data(url, None, self.http._headers_with_access_token())