def clear_min_string_length(self):
        """stub"""
        if (self.get_min_string_length_metadata().is_read_only() or
                self.get_min_string_length_metadata().is_required()):
            raise NoAccess()
        self.my_osid_object_form._my_map['minStringLength'] = \
            self.get_min_string_length_metadata().get_default_cardinal_values()[0]