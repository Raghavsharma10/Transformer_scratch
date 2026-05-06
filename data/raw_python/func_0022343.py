def add_weatherdata(self, data):
        """Appends weather data.

        Args:
            data (WeatherData): weather data object

        """
        if not isinstance(data, WeatherData):
            raise ValueError('Weather data need to be of type WeatherData')
        self._data["WEATHER DATA"].append(data)