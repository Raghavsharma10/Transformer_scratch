def mac_address_table_static_interface_type(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        mac_address_table = ET.SubElement(config, "mac-address-table", xmlns="urn:brocade.com:mgmt:brocade-mac-address-table")
        static = ET.SubElement(mac_address_table, "static")
        mac_address_key = ET.SubElement(static, "mac-address")
        mac_address_key.text = kwargs.pop('mac_address')
        forward_key = ET.SubElement(static, "forward")
        forward_key.text = kwargs.pop('forward')
        interface_name_key = ET.SubElement(static, "interface-name")
        interface_name_key.text = kwargs.pop('interface_name')
        vlan_key = ET.SubElement(static, "vlan")
        vlan_key.text = kwargs.pop('vlan')
        vlanid_key = ET.SubElement(static, "vlanid")
        vlanid_key.text = kwargs.pop('vlanid')
        interface_type = ET.SubElement(static, "interface-type")
        interface_type.text = kwargs.pop('interface_type')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)