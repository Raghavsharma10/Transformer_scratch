def show_support_save_status_output_show_support_save_status_status(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        show_support_save_status = ET.Element("show_support_save_status")
        config = show_support_save_status
        output = ET.SubElement(show_support_save_status, "output")
        show_support_save_status = ET.SubElement(output, "show-support-save-status")
        status = ET.SubElement(show_support_save_status, "status")
        status.text = kwargs.pop('status')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)