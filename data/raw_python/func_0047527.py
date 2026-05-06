def clear_first_angle_projection(self):
        """stub"""
        if (self.get_first_angle_projection_metadata().is_read_only() or
                self.get_first_angle_projection_metadata().is_required()):
            raise NoAccess()
        self.my_osid_object_form._my_map['firstAngle'] = \
            self._first_angle_metadata['default_boolean_values'][0]