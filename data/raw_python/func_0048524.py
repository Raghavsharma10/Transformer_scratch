def clear_spatial_unit_conditions(self):
        """stub"""
        if (self.get_spatial_unit_conditions_metadata().is_read_only() or
                self.get_zone_conditions_metadata().is_required()):
            raise NoAccess()
        self.my_osid_object_form._my_map['spatialUnitConditions'] = \
            self._zone_conditions_metadata['default_object_values'][0]