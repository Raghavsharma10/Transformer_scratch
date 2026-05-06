def system_monitor_mail_sfp_email_list_email(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        system_monitor_mail = ET.SubElement(config, "system-monitor-mail", xmlns="urn:brocade.com:mgmt:brocade-system-monitor")
        sfp = ET.SubElement(system_monitor_mail, "sfp")
        email_list = ET.SubElement(sfp, "email-list")
        email = ET.SubElement(email_list, "email")
        email.text = kwargs.pop('email')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)