def vcenter_discovery_ignore_delete_all_response_always(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        vcenter = ET.SubElement(config, "vcenter", xmlns="urn:brocade.com:mgmt:brocade-vswitch")
        id_key = ET.SubElement(vcenter, "id")
        id_key.text = kwargs.pop('id')
        discovery = ET.SubElement(vcenter, "discovery")
        ignore_delete_all_response = ET.SubElement(discovery, "ignore-delete-all-response")
        always = ET.SubElement(ignore_delete_all_response, "always")

        callback = kwargs.pop('callback', self._callback)
        return callback(config)