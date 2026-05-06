def mac_access_list_standard_hide_mac_acl_std_seq_action(self, **kwargs):
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
        seq_id_key = ET.SubElement(seq, "seq-id")
        seq_id_key.text = kwargs.pop('seq_id')
        action = ET.SubElement(seq, "action")
        action.text = kwargs.pop('action')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)