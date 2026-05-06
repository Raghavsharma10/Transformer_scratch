def clear_coordinate_conditions(self):
        """stub"""
        if (self.get_zone_conditions_metadata().is_read_only() or
                self.get_zone_conditions_metadata().is_required()):
            raise NoAccess()
        self.my_osid_object_form._my_map['coordinateConditions'] = \
            self._coordinate_conditions_metadata['default_object_values'][0]