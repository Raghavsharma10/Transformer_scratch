def coerce(self, value):
        """
        Coerces value to location hash.
        """

        return {
            'lat': float(value.get('lat', value.get('latitude'))),
            'lon': float(value.get('lon', value.get('longitude')))
        }