def system_monitor_MM_threshold_down_threshold(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        system_monitor = ET.SubElement(config, "system-monitor", xmlns="urn:brocade.com:mgmt:brocade-system-monitor")
        MM = ET.SubElement(system_monitor, "MM")
        threshold = ET.SubElement(MM, "threshold")
        down_threshold = ET.SubElement(threshold, "down-threshold")
        down_threshold.text = kwargs.pop('down_threshold')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)