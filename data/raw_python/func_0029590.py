def mac_address_table_static_forward(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        mac_address_table = ET.SubElement(config, "mac-address-table", xmlns="urn:brocade.com:mgmt:brocade-mac-address-table")
        static = ET.SubElement(mac_address_table, "static")
        mac_address_key = ET.SubElement(static, "mac-address")
        mac_address_key.text = kwargs.pop('mac_address')
        interface_type_key = ET.SubElement(static, "interface-type")
        interface_type_key.text = kwargs.pop('interface_type')
        interface_name_key = ET.SubElement(static, "interface-name")
        interface_name_key.text = kwargs.pop('interface_name')
        vlan_key = ET.SubElement(static, "vlan")
        vlan_key.text = kwargs.pop('vlan')
        vlanid_key = ET.SubElement(static, "vlanid")
        vlanid_key.text = kwargs.pop('vlanid')
        forward = ET.SubElement(static, "forward")
        forward.text = kwargs.pop('forward')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)