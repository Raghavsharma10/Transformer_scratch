def system_monitor_compact_flash_threshold_marginal_threshold(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        system_monitor = ET.SubElement(config, "system-monitor", xmlns="urn:brocade.com:mgmt:brocade-system-monitor")
        compact_flash = ET.SubElement(system_monitor, "compact-flash")
        threshold = ET.SubElement(compact_flash, "threshold")
        marginal_threshold = ET.SubElement(threshold, "marginal-threshold")
        marginal_threshold.text = kwargs.pop('marginal_threshold')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)