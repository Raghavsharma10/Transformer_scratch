def netconf_state_statistics_out_rpc_errors(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        netconf_state = ET.SubElement(config, "netconf-state", xmlns="urn:ietf:params:xml:ns:yang:ietf-netconf-monitoring")
        statistics = ET.SubElement(netconf_state, "statistics")
        out_rpc_errors = ET.SubElement(statistics, "out-rpc-errors")
        out_rpc_errors.text = kwargs.pop('out_rpc_errors')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)