def system_monitor_power_alert_action(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        system_monitor = ET.SubElement(config, "system-monitor", xmlns="urn:brocade.com:mgmt:brocade-system-monitor")
        power = ET.SubElement(system_monitor, "power")
        alert = ET.SubElement(power, "alert")
        action = ET.SubElement(alert, "action")
        action.text = kwargs.pop('action')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)