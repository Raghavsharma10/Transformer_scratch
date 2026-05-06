def set_first_angle_projection(self, value=None):
        """stub"""
        if value is None:
            raise NullArgument()
        if self.get_first_angle_projection_metadata().is_read_only():
            raise NoAccess()
        if not self.my_osid_object_form._is_valid_boolean(value):
            raise InvalidArgument()
        self.my_osid_object_form._my_map['firstAngle'] = value