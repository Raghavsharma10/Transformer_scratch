def get_cached_source_variable(self, source_id, variable, default=None):
        """ Get the cached value of a source variable. If the variable is not
        cached return the default value. """

        source_id = int(source_id)
        try:
            return self._retrieve_cached_source_variable(
                    source_id, variable)
        except UncachedVariable:
            return default