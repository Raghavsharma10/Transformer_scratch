def _realGetInstalledThemes(self, store):
        """
        Collect themes from all offerings installed on this store.
        """
        l = []
        for offering in getInstalledOfferings(store).itervalues():
            l.extend(offering.themes)
        l.sort(key=lambda o: o.priority)
        l.reverse()
        return l