def get_schema_input_version(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_schema = ET.Element("get_schema")
        config = get_schema
        input = ET.SubElement(get_schema, "input")
        version = ET.SubElement(input, "version")
        version.text = kwargs.pop('version')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)