def system_monitor_power_alert_state(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        system_monitor = ET.SubElement(config, "system-monitor", xmlns="urn:brocade.com:mgmt:brocade-system-monitor")
        power = ET.SubElement(system_monitor, "power")
        alert = ET.SubElement(power, "alert")
        state = ET.SubElement(alert, "state")
        state.text = kwargs.pop('state')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)