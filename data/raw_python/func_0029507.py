def show_support_save_status_output_show_support_save_status_percentage_of_completion(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        show_support_save_status = ET.Element("show_support_save_status")
        config = show_support_save_status
        output = ET.SubElement(show_support_save_status, "output")
        show_support_save_status = ET.SubElement(output, "show-support-save-status")
        percentage_of_completion = ET.SubElement(show_support_save_status, "percentage-of-completion")
        percentage_of_completion.text = kwargs.pop('percentage_of_completion')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)