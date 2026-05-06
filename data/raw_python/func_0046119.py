def _update_object_map(self, obj_map):
        """stub"""
        if self.has_color_coordinate() and \
                self.get_color_coordinate().get_coordinate_type() == RGB_COLOR_COORDINATE:
            obj_map['colorCoordinate']['hexValue'] = str(self.get_color_coordinate())
        try:
            super(ColorCoordinateRecord, self)._update_object_map(obj_map)
        except AttributeError:
            pass