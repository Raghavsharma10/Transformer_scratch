def clear_difficulty(self):
        """stub"""
        if (self.get_difficulty_metadata().is_read_only() or
                self.get_difficulty_metadata().is_required()):
            raise NoAccess()
        self.my_osid_object_form._my_map['texts']['difficulty'] = \
            self._difficulty_metadata['default_string_values'][0]