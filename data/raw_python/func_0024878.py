def get_weather(self):
        """
        Returns an instance of the Weather Service.
        """
        import predix.data.weather
        weather = predix.data.weather.WeatherForecast()
        return weather