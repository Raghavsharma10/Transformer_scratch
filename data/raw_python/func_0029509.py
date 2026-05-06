def show_system_info_output_show_system_info_rbridge_id(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        show_system_info = ET.Element("show_system_info")
        config = show_system_info
        output = ET.SubElement(show_system_info, "output")
        show_system_info = ET.SubElement(output, "show-system-info")
        rbridge_id = ET.SubElement(show_system_info, "rbridge-id")
        rbridge_id.text = kwargs.pop('rbridge_id')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)