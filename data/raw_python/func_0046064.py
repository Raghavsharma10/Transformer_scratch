def clear_text(self):
        """stub"""
        if (self.get_text_metadata().is_read_only() or
                self.get_text_metadata().is_required()):
            raise NoAccess()
        self.my_osid_object_form._my_map['text'] = \
            dict(self.get_text_metadata().get_default_string_values()[0])