def netconf_state_schemas_schema_identifier(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        netconf_state = ET.SubElement(config, "netconf-state", xmlns="urn:ietf:params:xml:ns:yang:ietf-netconf-monitoring")
        schemas = ET.SubElement(netconf_state, "schemas")
        schema = ET.SubElement(schemas, "schema")
        version_key = ET.SubElement(schema, "version")
        version_key.text = kwargs.pop('version')
        format_key = ET.SubElement(schema, "format")
        format_key.text = kwargs.pop('format')
        identifier = ET.SubElement(schema, "identifier")
        identifier.text = kwargs.pop('identifier')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)