def add_zone_condition(self, droppable_id, zone_id, match=True):
        """stub"""
        self.my_osid_object_form._my_map['zoneConditions'].append(
            {'droppableId': droppable_id, 'zoneId': zone_id, 'match': match})
        self.my_osid_object_form._my_map['zoneConditions'].sort(key=lambda k: k['zoneId'])