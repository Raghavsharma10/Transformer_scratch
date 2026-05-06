def get_source_variable(self, source_id, variable):
        """ Get the current value of a source variable. If the variable is not
        in the cache it will be retrieved from the controller. """

        source_id = int(source_id)
        try:
            return self._retrieve_cached_source_variable(
                    source_id, variable)
        except UncachedVariable:
            return (yield from self._send_cmd("GET S[%d].%s" % (
                source_id, variable)))