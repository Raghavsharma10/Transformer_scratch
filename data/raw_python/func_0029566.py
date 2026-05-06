def system_monitor_LineCard_threshold_down_threshold(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        system_monitor = ET.SubElement(config, "system-monitor", xmlns="urn:brocade.com:mgmt:brocade-system-monitor")
        LineCard = ET.SubElement(system_monitor, "LineCard")
        threshold = ET.SubElement(LineCard, "threshold")
        down_threshold = ET.SubElement(threshold, "down-threshold")
        down_threshold.text = kwargs.pop('down_threshold')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)