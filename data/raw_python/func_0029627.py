def policy_map_class_scheduler_strict_priority_priority_number(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        policy_map = ET.SubElement(config, "policy-map", xmlns="urn:brocade.com:mgmt:brocade-policer")
        po_name_key = ET.SubElement(policy_map, "po-name")
        po_name_key.text = kwargs.pop('po_name')
        class_el = ET.SubElement(policy_map, "class")
        cl_name_key = ET.SubElement(class_el, "cl-name")
        cl_name_key.text = kwargs.pop('cl_name')
        scheduler = ET.SubElement(class_el, "scheduler")
        strict_priority = ET.SubElement(scheduler, "strict-priority")
        priority_number = ET.SubElement(strict_priority, "priority-number")
        priority_number.text = kwargs.pop('priority_number')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)