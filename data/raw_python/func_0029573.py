def system_monitor_mail_sfp_enable(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        system_monitor_mail = ET.SubElement(config, "system-monitor-mail", xmlns="urn:brocade.com:mgmt:brocade-system-monitor")
        sfp = ET.SubElement(system_monitor_mail, "sfp")
        enable = ET.SubElement(sfp, "enable")

        callback = kwargs.pop('callback', self._callback)
        return callback(config)