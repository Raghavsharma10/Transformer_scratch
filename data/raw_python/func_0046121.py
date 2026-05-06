def set_color_coordinate(self, coordinate=None):
        """stub"""
        if coordinate is None:
            raise NullArgument()
        if self.get_color_coordinate_metadata().is_read_only():
            raise NoAccess()
        if not isinstance(coordinate, RGBColorCoordinate):
            raise InvalidArgument('coordinate must be instance of RGBColorCoordinate')
        self.my_osid_object_form._my_map['colorCoordinate']['values'] = \
            coordinate.get_values()
        self.my_osid_object_form._my_map['colorCoordinate']['uncertaintyPlus'] = \
            coordinate.get_uncertainty_plus()
        self.my_osid_object_form._my_map['colorCoordinate']['uncertaintyMinus'] = \
            coordinate.get_uncertainty_minus()