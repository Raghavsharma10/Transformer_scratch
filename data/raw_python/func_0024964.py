def create_weather(self, **kwargs):
        """
        Creates an instance of the Asset Service.
        """
        weather = predix.admin.weather.WeatherForecast(**kwargs)
        weather.create()

        client_id = self.get_client_id()
        if client_id:
            weather.grant_client(client_id)

        weather.grant_client(client_id)
        weather.add_to_manifest(self)
        return weather