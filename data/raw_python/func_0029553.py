def system_monitor_temp_threshold_marginal_threshold(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        system_monitor = ET.SubElement(config, "system-monitor", xmlns="urn:brocade.com:mgmt:brocade-system-monitor")
        temp = ET.SubElement(system_monitor, "temp")
        threshold = ET.SubElement(temp, "threshold")
        marginal_threshold = ET.SubElement(threshold, "marginal-threshold")
        marginal_threshold.text = kwargs.pop('marginal_threshold')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)