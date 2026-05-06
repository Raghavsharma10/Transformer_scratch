def fcoe_fcoe_fcf_map_fcf_map_fcoe_map_fcf_map_fcoe_map_leaf(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        fcoe = ET.SubElement(config, "fcoe", xmlns="urn:brocade.com:mgmt:brocade-fcoe")
        fcoe_fcf_map = ET.SubElement(fcoe, "fcoe-fcf-map")
        fcf_map_name_key = ET.SubElement(fcoe_fcf_map, "fcf-map-name")
        fcf_map_name_key.text = kwargs.pop('fcf_map_name')
        fcf_map_fcoe_map = ET.SubElement(fcoe_fcf_map, "fcf-map-fcoe-map")
        fcf_map_fcoe_map_leaf = ET.SubElement(fcf_map_fcoe_map, "fcf-map-fcoe-map-leaf")
        fcf_map_fcoe_map_leaf.text = kwargs.pop('fcf_map_fcoe_map_leaf')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)