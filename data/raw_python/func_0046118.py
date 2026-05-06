def get_color_coordinate(self):
        """stub"""
        if self.has_color_coordinate():
            color_dict = self.my_osid_object._my_map['colorCoordinate']
            return RGBColorCoordinate(values=color_dict['values'],
                                      uncertainty_minus=color_dict['uncertaintyMinus'],
                                      uncertainty_plus=color_dict['uncertaintyPlus'])
        raise IllegalState('No color coordinate')