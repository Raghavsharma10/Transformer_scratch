def get_cached_zone_variable(self, zone_id, variable, default=None):
        """ Retrieve the current value of a zone variable from the cache or
        return the default value if the variable is not present. """

        try:
            return self._retrieve_cached_zone_variable(zone_id, variable)
        except UncachedVariable:
            return default