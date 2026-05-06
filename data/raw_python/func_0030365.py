def dotted(self):
        """Return just the tract number, excluding the state and county, in the dotted format"""
        v = str(self.geoid.tract).zfill(6)
        return v[0:4] + '.' + v[4:]