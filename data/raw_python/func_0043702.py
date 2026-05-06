def _retrieve_cached_source_variable(self, source_id, name):
        """
        Retrieves the cache state of the named variable for a particular
        source. If the variable has not been cached then the UncachedVariable
        exception is raised.
        """
        try:
            s = self._source_state[source_id][name.lower()]
            logger.debug("Source Cache retrieve S[%d].%s = %s",
                         source_id, name, s)
            return s
        except KeyError:
            raise UncachedVariable