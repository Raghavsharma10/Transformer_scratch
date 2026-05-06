def system_monitor_mail_fru_email_list_email(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        system_monitor_mail = ET.SubElement(config, "system-monitor-mail", xmlns="urn:brocade.com:mgmt:brocade-system-monitor")
        fru = ET.SubElement(system_monitor_mail, "fru")
        email_list = ET.SubElement(fru, "email-list")
        email = ET.SubElement(email_list, "email")
        email.text = kwargs.pop('email')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)