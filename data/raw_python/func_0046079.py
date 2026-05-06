def clear_integer_values(self):
        """stub"""
        if self._integer_values_metadata['required'] or \
                self._integer_values_metadata['read_only']:
            raise NoAccess()
        self.my_osid_object_form._my_map['integerValues'] = \
            dict(self._integer_values_metadata['default_object_values'][0])