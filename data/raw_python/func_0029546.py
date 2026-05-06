def system_monitor_fan_threshold_down_threshold(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        system_monitor = ET.SubElement(config, "system-monitor", xmlns="urn:brocade.com:mgmt:brocade-system-monitor")
        fan = ET.SubElement(system_monitor, "fan")
        threshold = ET.SubElement(fan, "threshold")
        down_threshold = ET.SubElement(threshold, "down-threshold")
        down_threshold.text = kwargs.pop('down_threshold')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)