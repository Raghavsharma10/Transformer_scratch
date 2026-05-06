def show_raslog_output_show_all_raslog_raslog_entries_date_and_time_info(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        show_raslog = ET.Element("show_raslog")
        config = show_raslog
        output = ET.SubElement(show_raslog, "output")
        show_all_raslog = ET.SubElement(output, "show-all-raslog")
        raslog_entries = ET.SubElement(show_all_raslog, "raslog-entries")
        date_and_time_info = ET.SubElement(raslog_entries, "date-and-time-info")
        date_and_time_info.text = kwargs.pop('date_and_time_info')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)