def _retrieve_cached_zone_variable(self, zone_id, name):
        """
        Retrieves the cache state of the named variable for a particular
        zone. If the variable has not been cached then the UncachedVariable
        exception is raised.
        """
        try:
            s = self._zone_state[zone_id][name.lower()]
            logger.debug("Zone Cache retrieve %s.%s = %s",
                         zone_id.device_str(), name, s)
            return s
        except KeyError:
            raise UncachedVariable