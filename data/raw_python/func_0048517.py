def _init_map(self):
        """stub"""
        self.my_osid_object_form._my_map['zoneConditions'] = \
            self._zone_conditions_metadata['default_object_values'][0]
        self.my_osid_object_form._my_map['coordinateConditions'] = \
            self._coordinate_conditions_metadata['default_object_values'][0]
        self.my_osid_object_form._my_map['spatialUnitConditions'] = \
            self._spatial_unit_conditions_metadata['default_object_values'][0]