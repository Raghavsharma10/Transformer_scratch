def get_zone_variable(self, zone_id, variable):
        """ Retrieve the current value of a zone variable.  If the variable is
        not found in the local cache then the value is requested from the
        controller.  """

        try:
            return self._retrieve_cached_zone_variable(zone_id, variable)
        except UncachedVariable:
            return (yield from self._send_cmd("GET %s.%s" % (
                zone_id.device_str(), variable)))