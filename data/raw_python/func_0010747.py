def parent(self):
        """Return a location representing the administrative unit above
        the one represented by this location."""
        if self.city:
            return Location(
                country=self.country, state=self.state, county=self.county)
        if self.county:
            return Location(country=self.country, state=self.state)
        if self.state:
            return Location(country=self.country)
        return Location()