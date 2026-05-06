def canonical(self):
        """Return a tuple containing a canonicalized version of this
        location's country, state, county, and city names."""
        try:
            return tuple(map(lambda x: x.lower(), self.name()))
        except:
            return tuple([x.lower() for x in self.name()])