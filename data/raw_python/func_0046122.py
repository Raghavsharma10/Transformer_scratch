def clear_color_coordinate(self):
        """stub"""
        if (self.get_color_coordinate_metadata().is_read_only() or
                self.get_color_coordinate_metadata().is_required()):
            raise NoAccess()
        self.my_osid_object_form._my_map['colorCoordinate'] = \
            dict(self.get_color_coordinate_metadata().get_default_coordinate_values()[0])