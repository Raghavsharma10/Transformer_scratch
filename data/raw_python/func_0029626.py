def policy_map_class_police_eir(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        policy_map = ET.SubElement(config, "policy-map", xmlns="urn:brocade.com:mgmt:brocade-policer")
        po_name_key = ET.SubElement(policy_map, "po-name")
        po_name_key.text = kwargs.pop('po_name')
        class_el = ET.SubElement(policy_map, "class")
        cl_name_key = ET.SubElement(class_el, "cl-name")
        cl_name_key.text = kwargs.pop('cl_name')
        police = ET.SubElement(class_el, "police")
        eir = ET.SubElement(police, "eir")
        eir.text = kwargs.pop('eir')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)