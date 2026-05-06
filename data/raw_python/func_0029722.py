def fcoe_fcoe_fcf_map_fcf_map_name(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        fcoe = ET.SubElement(config, "fcoe", xmlns="urn:brocade.com:mgmt:brocade-fcoe")
        fcoe_fcf_map = ET.SubElement(fcoe, "fcoe-fcf-map")
        fcf_map_name = ET.SubElement(fcoe_fcf_map, "fcf-map-name")
        fcf_map_name.text = kwargs.pop('fcf_map_name')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)