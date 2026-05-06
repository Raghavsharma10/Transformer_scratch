def show_support_save_status_output_show_support_save_status_rbridge_id(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        show_support_save_status = ET.Element("show_support_save_status")
        config = show_support_save_status
        output = ET.SubElement(show_support_save_status, "output")
        show_support_save_status = ET.SubElement(output, "show-support-save-status")
        rbridge_id = ET.SubElement(show_support_save_status, "rbridge-id")
        rbridge_id.text = kwargs.pop('rbridge_id')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)