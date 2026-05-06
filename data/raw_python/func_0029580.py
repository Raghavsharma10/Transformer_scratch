def system_monitor_mail_relay_domain_name(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        system_monitor_mail = ET.SubElement(config, "system-monitor-mail", xmlns="urn:brocade.com:mgmt:brocade-system-monitor")
        relay = ET.SubElement(system_monitor_mail, "relay")
        host_ip_key = ET.SubElement(relay, "host-ip")
        host_ip_key.text = kwargs.pop('host_ip')
        domain_name = ET.SubElement(relay, "domain-name")
        domain_name.text = kwargs.pop('domain_name')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)