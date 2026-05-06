def get_mac_acl_for_intf_output_interface_ingress_policy_policy_name(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_mac_acl_for_intf = ET.Element("get_mac_acl_for_intf")
        config = get_mac_acl_for_intf
        output = ET.SubElement(get_mac_acl_for_intf, "output")
        interface = ET.SubElement(output, "interface")
        interface_type_key = ET.SubElement(interface, "interface-type")
        interface_type_key.text = kwargs.pop('interface_type')
        interface_name_key = ET.SubElement(interface, "interface-name")
        interface_name_key.text = kwargs.pop('interface_name')
        ingress_policy = ET.SubElement(interface, "ingress-policy")
        policy_name = ET.SubElement(ingress_policy, "policy-name")
        policy_name.text = kwargs.pop('policy_name')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)