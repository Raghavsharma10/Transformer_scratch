def _store_cached_source_variable(self, source_id, name, value):
        """
        Stores the current known value of a source variable into the cache.
        Calls any source callbacks.
        """
        source_state = self._source_state.setdefault(source_id, {})
        name = name.lower()
        source_state[name] = value
        logger.debug("Source Cache store S[%d].%s = %s",
                     source_id, name, value)
        for callback in self._source_callbacks:
            callback(source_id, name, value)