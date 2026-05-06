def get_weather_in(self, place, unit=None, items=None):
        """Return weather info according to place
        """
        unit = unit if unit else self.unit
        response = self.select('weather.forecast', items=items).where(['woeid','IN',('SELECT woeid FROM geo.places WHERE text="{0}"'.format(place),)], ['u','=',unit] if unit else [])
        return response