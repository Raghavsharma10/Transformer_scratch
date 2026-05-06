def get_system_uptime_output_show_system_uptime_days(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_system_uptime = ET.Element("get_system_uptime")
        config = get_system_uptime
        output = ET.SubElement(get_system_uptime, "output")
        show_system_uptime = ET.SubElement(output, "show-system-uptime")
        rbridge_id_key = ET.SubElement(show_system_uptime, "rbridge-id")
        rbridge_id_key.text = kwargs.pop('rbridge_id')
        days = ET.SubElement(show_system_uptime, "days")
        days.text = kwargs.pop('days')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)