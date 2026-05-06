def get_schema_input_identifier(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_schema = ET.Element("get_schema")
        config = get_schema
        input = ET.SubElement(get_schema, "input")
        identifier = ET.SubElement(input, "identifier")
        identifier.text = kwargs.pop('identifier')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)