def fetch_mood_station(self, station_id, terr=KKBOXTerritory.TAIWAN):
        '''
        Fetches a mood station by given ID.

        :param station_id: the station ID
        :param terr: the current territory.
        :return: API response.
        :rtype: dict

        See `https://docs-en.kkbox.codes/v1.1/reference#moodstations-station_id`.
        '''
        url = 'https://api.kkbox.com/v1.1/mood-stations/%s' % station_id
        url += '?' + url_parse.urlencode({'territory': terr})
        return self.http._post_data(url, None, self.http._headers_with_access_token())