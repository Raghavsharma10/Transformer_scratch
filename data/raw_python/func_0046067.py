def clear_integer_value(self):
        """stub"""
        if (self.get_integer_value_metadata().is_read_only() or
                self.get_integer_value_metadata().is_required()):
            raise NoAccess()
        self.my_osid_object_form._my_map['integerValue'] = \
            self.get_integer_value_metadata().get_default_integer_values()[0]