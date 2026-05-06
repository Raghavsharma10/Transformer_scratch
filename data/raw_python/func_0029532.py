def netconf_state_statistics_in_sessions(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        netconf_state = ET.SubElement(config, "netconf-state", xmlns="urn:ietf:params:xml:ns:yang:ietf-netconf-monitoring")
        statistics = ET.SubElement(netconf_state, "statistics")
        in_sessions = ET.SubElement(statistics, "in-sessions")
        in_sessions.text = kwargs.pop('in_sessions')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)