def mac_access_list_extended_hide_mac_acl_ext_seq_seq_id(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        mac = ET.SubElement(config, "mac", xmlns="urn:brocade.com:mgmt:brocade-mac-access-list")
        access_list = ET.SubElement(mac, "access-list")
        extended = ET.SubElement(access_list, "extended")
        name_key = ET.SubElement(extended, "name")
        name_key.text = kwargs.pop('name')
        hide_mac_acl_ext = ET.SubElement(extended, "hide-mac-acl-ext")
        seq = ET.SubElement(hide_mac_acl_ext, "seq")
        seq_id = ET.SubElement(seq, "seq-id")
        seq_id.text = kwargs.pop('seq_id')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)