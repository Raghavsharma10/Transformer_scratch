def show_raslog_output_show_all_raslog_raslog_entries_switch_or_chassis_name(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        show_raslog = ET.Element("show_raslog")
        config = show_raslog
        output = ET.SubElement(show_raslog, "output")
        show_all_raslog = ET.SubElement(output, "show-all-raslog")
        raslog_entries = ET.SubElement(show_all_raslog, "raslog-entries")
        switch_or_chassis_name = ET.SubElement(raslog_entries, "switch-or-chassis-name")
        switch_or_chassis_name.text = kwargs.pop('switch_or_chassis_name')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)