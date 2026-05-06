def system_monitor_cid_card_alert_action(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        system_monitor = ET.SubElement(config, "system-monitor", xmlns="urn:brocade.com:mgmt:brocade-system-monitor")
        cid_card = ET.SubElement(system_monitor, "cid-card")
        alert = ET.SubElement(cid_card, "alert")
        action = ET.SubElement(alert, "action")
        action.text = kwargs.pop('action')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)