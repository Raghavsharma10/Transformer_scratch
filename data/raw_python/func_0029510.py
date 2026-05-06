def show_system_info_output_show_system_info_stack_mac(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        show_system_info = ET.Element("show_system_info")
        config = show_system_info
        output = ET.SubElement(show_system_info, "output")
        show_system_info = ET.SubElement(output, "show-system-info")
        stack_mac = ET.SubElement(show_system_info, "stack-mac")
        stack_mac.text = kwargs.pop('stack_mac')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)