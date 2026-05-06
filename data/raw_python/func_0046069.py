def clear_decimal_value(self):
        """stub"""
        if (self.get_decimal_value_metadata().is_read_only() or
                self.get_decimal_value_metadata().is_required()):
            raise NoAccess()
        self.my_osid_object_form._my_map['decimalValue'] = \
            self.get_decimal_value_metadata().get_default_decimal_values()[0]