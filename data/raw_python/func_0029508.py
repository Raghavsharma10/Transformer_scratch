def show_system_info_input_rbridge_id(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        show_system_info = ET.Element("show_system_info")
        config = show_system_info
        input = ET.SubElement(show_system_info, "input")
        rbridge_id = ET.SubElement(input, "rbridge-id")
        rbridge_id.text = kwargs.pop('rbridge_id')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)