def system_monitor_mail_interface_enable(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        system_monitor_mail = ET.SubElement(config, "system-monitor-mail", xmlns="urn:brocade.com:mgmt:brocade-system-monitor")
        interface = ET.SubElement(system_monitor_mail, "interface")
        enable = ET.SubElement(interface, "enable")

        callback = kwargs.pop('callback', self._callback)
        return callback(config)