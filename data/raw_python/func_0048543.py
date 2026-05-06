def remove_zone(self, zone_id):
        """remove a zone, given the id"""
        updated_zones = []
        for zone in self.my_osid_object_form._my_map['zones']:
            if zone['id'] != zone_id:
                updated_zones.append(zone)
        self.my_osid_object_form._my_map['zones'] = updated_zones