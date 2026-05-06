def _parse_forecasts(self):
        """
        Returns a list of instances of the forecast.Forecast class. Each
        instance of the class is instantiated with the attributes of the
        forecast elements in the RSS feed.
        """
        forecasts = self._channel.findall(
            './/{0}{1}'.format(self.weather_namespace, 'forecast')
        )
        return [Forecast(forecast.attrib) for forecast in forecasts]