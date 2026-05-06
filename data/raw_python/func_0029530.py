def netconf_state_statistics_netconf_start_time(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        netconf_state = ET.SubElement(config, "netconf-state", xmlns="urn:ietf:params:xml:ns:yang:ietf-netconf-monitoring")
        statistics = ET.SubElement(netconf_state, "statistics")
        netconf_start_time = ET.SubElement(statistics, "netconf-start-time")
        netconf_start_time.text = kwargs.pop('netconf_start_time')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)