def strings(self):
        """ Return lat/lon as strings. """
        return [
            toString(self.lat, LAT),
            toString(self.lon, LON)
        ]