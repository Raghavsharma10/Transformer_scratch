def name(self):
        """Return a tuple containing this location's country, state,
        county, and city names."""
        try:
            return tuple(
                getattr(self, x) if getattr(self, x) else u''
                for x in ('country', 'state', 'county', 'city'))
        except:
            return tuple(
                getattr(self, x) if getattr(self, x) else ''
                for x in ('country', 'state', 'county', 'city'))