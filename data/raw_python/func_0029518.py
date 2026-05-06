def netconf_state_schemas_schema_version(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        netconf_state = ET.SubElement(config, "netconf-state", xmlns="urn:ietf:params:xml:ns:yang:ietf-netconf-monitoring")
        schemas = ET.SubElement(netconf_state, "schemas")
        schema = ET.SubElement(schemas, "schema")
        identifier_key = ET.SubElement(schema, "identifier")
        identifier_key.text = kwargs.pop('identifier')
        format_key = ET.SubElement(schema, "format")
        format_key.text = kwargs.pop('format')
        version = ET.SubElement(schema, "version")
        version.text = kwargs.pop('version')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)