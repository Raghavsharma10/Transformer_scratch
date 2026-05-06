def netconf_state_statistics_out_notifications(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        netconf_state = ET.SubElement(config, "netconf-state", xmlns="urn:ietf:params:xml:ns:yang:ietf-netconf-monitoring")
        statistics = ET.SubElement(netconf_state, "statistics")
        out_notifications = ET.SubElement(statistics, "out-notifications")
        out_notifications.text = kwargs.pop('out_notifications')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)