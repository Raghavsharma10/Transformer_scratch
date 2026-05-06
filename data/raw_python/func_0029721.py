def fcoe_fcoe_map_fcoe_map_fabric_map_fcoe_map_fabric_map_name(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        fcoe = ET.SubElement(config, "fcoe", xmlns="urn:brocade.com:mgmt:brocade-fcoe")
        fcoe_map = ET.SubElement(fcoe, "fcoe-map")
        fcoe_map_name_key = ET.SubElement(fcoe_map, "fcoe-map-name")
        fcoe_map_name_key.text = kwargs.pop('fcoe_map_name')
        fcoe_map_fabric_map = ET.SubElement(fcoe_map, "fcoe-map-fabric-map")
        fcoe_map_fabric_map_name = ET.SubElement(fcoe_map_fabric_map, "fcoe-map-fabric-map-name")
        fcoe_map_fabric_map_name.text = kwargs.pop('fcoe_map_fabric_map_name')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)