def system_monitor_MM_threshold_marginal_threshold(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        system_monitor = ET.SubElement(config, "system-monitor", xmlns="urn:brocade.com:mgmt:brocade-system-monitor")
        MM = ET.SubElement(system_monitor, "MM")
        threshold = ET.SubElement(MM, "threshold")
        marginal_threshold = ET.SubElement(threshold, "marginal-threshold")
        marginal_threshold.text = kwargs.pop('marginal_threshold')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)