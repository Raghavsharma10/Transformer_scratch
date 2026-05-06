def _realGetAllThemes(self):
        """
        Collect themes from all available offerings.
        """
        l = []
        for offering in getOfferings():
            l.extend(offering.themes)
        l.sort(key=lambda o: o.priority)
        l.reverse()
        return l