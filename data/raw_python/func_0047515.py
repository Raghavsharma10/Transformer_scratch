def clear_max_string_length(self):
        """stub"""
        if (self.get_max_string_length_metadata().is_read_only() or
                self.get_max_string_length_metadata().is_required()):
            raise NoAccess()
        self.my_osid_object_form._my_map['maxStringLength'] = \
            self.get_max_string_length_metadata().get_default_cardinal_values()[0]