def get_system_uptime_output_show_system_uptime_rbridge_id(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_system_uptime = ET.Element("get_system_uptime")
        config = get_system_uptime
        output = ET.SubElement(get_system_uptime, "output")
        show_system_uptime = ET.SubElement(output, "show-system-uptime")
        rbridge_id = ET.SubElement(show_system_uptime, "rbridge-id")
        rbridge_id.text = kwargs.pop('rbridge_id')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)