def system_monitor_mail_relay_host_ip(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        system_monitor_mail = ET.SubElement(config, "system-monitor-mail", xmlns="urn:brocade.com:mgmt:brocade-system-monitor")
        relay = ET.SubElement(system_monitor_mail, "relay")
        host_ip = ET.SubElement(relay, "host-ip")
        host_ip.text = kwargs.pop('host_ip')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)