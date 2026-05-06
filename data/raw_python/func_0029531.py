def netconf_state_statistics_in_bad_hellos(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        netconf_state = ET.SubElement(config, "netconf-state", xmlns="urn:ietf:params:xml:ns:yang:ietf-netconf-monitoring")
        statistics = ET.SubElement(netconf_state, "statistics")
        in_bad_hellos = ET.SubElement(statistics, "in-bad-hellos")
        in_bad_hellos.text = kwargs.pop('in_bad_hellos')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)