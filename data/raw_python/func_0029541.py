def netconf_state_files_file_context(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        netconf_state = ET.SubElement(config, "netconf-state", xmlns="urn:ietf:params:xml:ns:yang:ietf-netconf-monitoring")
        files = ET.SubElement(netconf_state, "files", xmlns="http://tail-f.com/yang/netconf-monitoring")
        file = ET.SubElement(files, "file")
        name_key = ET.SubElement(file, "name")
        name_key.text = kwargs.pop('name')
        context = ET.SubElement(file, "context")
        context.text = kwargs.pop('context')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)