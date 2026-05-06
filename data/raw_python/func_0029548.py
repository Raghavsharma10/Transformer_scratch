def system_monitor_fan_alert_action(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        system_monitor = ET.SubElement(config, "system-monitor", xmlns="urn:brocade.com:mgmt:brocade-system-monitor")
        fan = ET.SubElement(system_monitor, "fan")
        alert = ET.SubElement(fan, "alert")
        action = ET.SubElement(alert, "action")
        action.text = kwargs.pop('action')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)