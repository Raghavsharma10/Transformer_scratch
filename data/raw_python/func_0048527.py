def get_zones(self):
        """stub"""
        zones = []
        for current_zone in self.my_osid_object._my_map['zones']:
            zones.append({
                'id': current_zone['id'],
                'name': self.get_matching_language_value('names',
                                                         dictionary=current_zone).text,
                'description': self.get_matching_language_value('descriptions',
                                                                dictionary=current_zone).text,
                'spatialUnit': SpatialUnitFactory().get_spatial_unit(current_zone['spatialUnit']).get_spatial_unit_map(),
                'containerId': current_zone['containerId'],
                'visible': current_zone['visible'],
                'reuse': current_zone['reuse'],
                'dropBehaviorType': current_zone['dropBehaviorType']
            })
        return zones