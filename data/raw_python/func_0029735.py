def mac_access_list_standard_hide_mac_acl_std_seq_seq_id(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        mac = ET.SubElement(config, "mac", xmlns="urn:brocade.com:mgmt:brocade-mac-access-list")
        access_list = ET.SubElement(mac, "access-list")
        standard = ET.SubElement(access_list, "standard")
        name_key = ET.SubElement(standard, "name")
        name_key.text = kwargs.pop('name')
        hide_mac_acl_std = ET.SubElement(standard, "hide-mac-acl-std")
        seq = ET.SubElement(hide_mac_acl_std, "seq")
        seq_id = ET.SubElement(seq, "seq-id")
        seq_id.text = kwargs.pop('seq_id')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)