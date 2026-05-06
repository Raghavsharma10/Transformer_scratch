def show_raslog_output_show_all_raslog_raslog_entries_message_flag(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        show_raslog = ET.Element("show_raslog")
        config = show_raslog
        output = ET.SubElement(show_raslog, "output")
        show_all_raslog = ET.SubElement(output, "show-all-raslog")
        raslog_entries = ET.SubElement(show_all_raslog, "raslog-entries")
        message_flag = ET.SubElement(raslog_entries, "message-flag")
        message_flag.text = kwargs.pop('message_flag')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)