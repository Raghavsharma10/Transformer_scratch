def system_monitor_LineCard_alert_state(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        system_monitor = ET.SubElement(config, "system-monitor", xmlns="urn:brocade.com:mgmt:brocade-system-monitor")
        LineCard = ET.SubElement(system_monitor, "LineCard")
        alert = ET.SubElement(LineCard, "alert")
        state = ET.SubElement(alert, "state")
        state.text = kwargs.pop('state')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)