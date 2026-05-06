def clear_choices(self):
        """stub"""
        if self.get_choices_metadata().is_read_only():
            raise NoAccess()
        self.my_osid_object_form._my_map['choices'] = \
            self._choices_metadata['default_object_values'][0]