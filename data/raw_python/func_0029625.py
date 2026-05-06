def policy_map_class_cl_name(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        policy_map = ET.SubElement(config, "policy-map", xmlns="urn:brocade.com:mgmt:brocade-policer")
        po_name_key = ET.SubElement(policy_map, "po-name")
        po_name_key.text = kwargs.pop('po_name')
        class_el = ET.SubElement(policy_map, "class_el")
        cl_name = ET.SubElement(class_el, "cl-name")
        cl_name.text = kwargs.pop('cl_name')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)