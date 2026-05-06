def system_monitor_mail_fru_enable(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        system_monitor_mail = ET.SubElement(config, "system-monitor-mail", xmlns="urn:brocade.com:mgmt:brocade-system-monitor")
        fru = ET.SubElement(system_monitor_mail, "fru")
        enable = ET.SubElement(fru, "enable")

        callback = kwargs.pop('callback', self._callback)
        return callback(config)