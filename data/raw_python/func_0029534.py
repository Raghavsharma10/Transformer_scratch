def netconf_state_statistics_in_rpcs(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        netconf_state = ET.SubElement(config, "netconf-state", xmlns="urn:ietf:params:xml:ns:yang:ietf-netconf-monitoring")
        statistics = ET.SubElement(netconf_state, "statistics")
        in_rpcs = ET.SubElement(statistics, "in-rpcs")
        in_rpcs.text = kwargs.pop('in_rpcs')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)