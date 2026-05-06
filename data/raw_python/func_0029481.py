def get_system_uptime_output_cmd_error(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_system_uptime = ET.Element("get_system_uptime")
        config = get_system_uptime
        output = ET.SubElement(get_system_uptime, "output")
        cmd_error = ET.SubElement(output, "cmd-error")
        cmd_error.text = kwargs.pop('cmd_error')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)