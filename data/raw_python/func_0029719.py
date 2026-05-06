def vcenter_discovery_ignore_delete_all_response_ignore_value(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        vcenter = ET.SubElement(config, "vcenter", xmlns="urn:brocade.com:mgmt:brocade-vswitch")
        id_key = ET.SubElement(vcenter, "id")
        id_key.text = kwargs.pop('id')
        discovery = ET.SubElement(vcenter, "discovery")
        ignore_delete_all_response = ET.SubElement(discovery, "ignore-delete-all-response")
        ignore_value = ET.SubElement(ignore_delete_all_response, "ignore-value")
        ignore_value.text = kwargs.pop('ignore_value')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)