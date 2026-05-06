def logging_syslog_server_use_vrf(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        logging = ET.SubElement(config, "logging", xmlns="urn:brocade.com:mgmt:brocade-ras")
        syslog_server = ET.SubElement(logging, "syslog-server")
        syslogip_key = ET.SubElement(syslog_server, "syslogip")
        syslogip_key.text = kwargs.pop('syslogip')
        use_vrf = ET.SubElement(syslog_server, "use-vrf")
        use_vrf.text = kwargs.pop('use_vrf')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)