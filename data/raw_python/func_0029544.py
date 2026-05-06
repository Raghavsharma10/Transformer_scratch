def get_schema_input_format(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_schema = ET.Element("get_schema")
        config = get_schema
        input = ET.SubElement(get_schema, "input")
        format = ET.SubElement(input, "format")
        format.text = kwargs.pop('format')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)