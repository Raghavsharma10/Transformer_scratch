def clear_decimal_values(self):
        """stub"""
        if self._decimal_values_metadata['required'] or \
                self._decimal_values_metadata['read_only']:
            raise NoAccess()
        self.my_osid_object_form._my_map['decimalValues'] = \
            dict(self._decimal_values_metadata['default_object_values'][0])