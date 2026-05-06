def show_raslog_output_show_all_raslog_rbridge_id(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        show_raslog = ET.Element("show_raslog")
        config = show_raslog
        output = ET.SubElement(show_raslog, "output")
        show_all_raslog = ET.SubElement(output, "show-all-raslog")
        rbridge_id = ET.SubElement(show_all_raslog, "rbridge-id")
        rbridge_id.text = kwargs.pop('rbridge_id')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)