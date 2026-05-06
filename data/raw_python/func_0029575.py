def system_monitor_mail_security_enable(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        system_monitor_mail = ET.SubElement(config, "system-monitor-mail", xmlns="urn:brocade.com:mgmt:brocade-system-monitor")
        security = ET.SubElement(system_monitor_mail, "security")
        enable = ET.SubElement(security, "enable")

        callback = kwargs.pop('callback', self._callback)
        return callback(config)