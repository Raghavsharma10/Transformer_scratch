def fetch_all_mood_stations(self, terr=KKBOXTerritory.TAIWAN):
        '''
        Fetches all mood stations.

        :param terr: the current territory.
        :return: API response.
        :rtype: dict

        See `https://docs-en.kkbox.codes/v1.1/reference#moodstations`.
        '''
        url = 'https://api.kkbox.com/v1.1/mood-stations'
        url += '?' + url_parse.urlencode({'territory': terr})
        return self.http._post_data(url, None, self.http._headers_with_access_token())