def get_weather_forecast_days(self, latitude, longitude,
            days=1, frequency=1, reading_type=None):
        """
        Return the weather forecast for a given location.

        ::

            results = ws.get_weather_forecast_days(lat, long)
            for w in results['hits']:
                print w['start_datetime_local']
                print w['reading_type'], w['reading_value']

        For description of reading types:
        https://graphical.weather.gov/xml/docs/elementInputNames.php
        """
        params = {}

        # Can get data from NWS1 or NWS3 representing 1-hr and 3-hr
        # intervals.
        if frequency not in [1, 3]:
            raise ValueError("Reading frequency must be 1 or 3")

        params['days'] = days
        params['source'] = 'NWS' + str(frequency)
        params['latitude'] = latitude
        params['longitude'] = longitude

        if reading_type:
            # url encoding will make spaces a + instead of %20, which service
            # interprets as an "and" search which is undesirable
            reading_type = reading_type.replace(' ', '%20')
            params['reading_type'] = urllib.quote_plus(reading_type)

        url = self.uri + '/v1/weather-forecast-days/'
        return self.service._get(url, params=params)