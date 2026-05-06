def set_zone_variable(self, zone_id, variable, value):
        """
        Set a zone variable to a new value.
        """
        return self._send_cmd("SET %s.%s=\"%s\"" % (
            zone_id.device_str(), variable, value))