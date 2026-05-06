def show_raslog_output_cmd_status_error_msg(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        show_raslog = ET.Element("show_raslog")
        config = show_raslog
        output = ET.SubElement(show_raslog, "output")
        cmd_status_error_msg = ET.SubElement(output, "cmd-status-error-msg")
        cmd_status_error_msg.text = kwargs.pop('cmd_status_error_msg')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)