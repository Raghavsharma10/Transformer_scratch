def netconf_state_files_file_created(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        netconf_state = ET.SubElement(config, "netconf-state", xmlns="urn:ietf:params:xml:ns:yang:ietf-netconf-monitoring")
        files = ET.SubElement(netconf_state, "files", xmlns="http://tail-f.com/yang/netconf-monitoring")
        file = ET.SubElement(files, "file")
        name_key = ET.SubElement(file, "name")
        name_key.text = kwargs.pop('name')
        created = ET.SubElement(file, "created")
        created.text = kwargs.pop('created')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)