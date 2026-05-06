def show_raslog_input_rbridge_id(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        show_raslog = ET.Element("show_raslog")
        config = show_raslog
        input = ET.SubElement(show_raslog, "input")
        rbridge_id = ET.SubElement(input, "rbridge-id")
        rbridge_id.text = kwargs.pop('rbridge_id')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)