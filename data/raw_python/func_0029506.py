def show_support_save_status_output_show_support_save_status_message(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        show_support_save_status = ET.Element("show_support_save_status")
        config = show_support_save_status
        output = ET.SubElement(show_support_save_status, "output")
        show_support_save_status = ET.SubElement(output, "show-support-save-status")
        message = ET.SubElement(show_support_save_status, "message")
        message.text = kwargs.pop('message')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)