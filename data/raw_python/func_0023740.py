def fetch_top_tracks_of_artist(self, artist_id, terr=KKBOXTerritory.TAIWAN):
        '''
        Fetcher top tracks belong to an artist by given ID.

        :param artist_id: the artist ID.
        :type artist_id: str
        :param terr: the current territory.
        :return: API response.
        :rtype: dict

        See 'https://docs-en.kkbox.codes/v1.1/reference#artists-artist_id-toptracks'
        '''
        url = 'https://api.kkbox.com/v1.1/artists/%s/top-tracks' % artist_id
        url += '?' + url_parse.urlencode({'territory': terr})
        return self.http._post_data(url, None, self.http._headers_with_access_token())