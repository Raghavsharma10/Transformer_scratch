def system_monitor_sfp_alert_action(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        system_monitor = ET.SubElement(config, "system-monitor", xmlns="urn:brocade.com:mgmt:brocade-system-monitor")
        sfp = ET.SubElement(system_monitor, "sfp")
        alert = ET.SubElement(sfp, "alert")
        action = ET.SubElement(alert, "action")
        action.text = kwargs.pop('action')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)