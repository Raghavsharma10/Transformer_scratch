def show_raslog_output_show_all_raslog_number_of_entries(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        show_raslog = ET.Element("show_raslog")
        config = show_raslog
        output = ET.SubElement(show_raslog, "output")
        show_all_raslog = ET.SubElement(output, "show-all-raslog")
        number_of_entries = ET.SubElement(show_all_raslog, "number-of-entries")
        number_of_entries.text = kwargs.pop('number_of_entries')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)