def clear_source(self):
        """stub"""
        if (self.get_source_metadata().is_read_only() or
                self.get_source_metadata().is_required()):
            raise NoAccess()
        self.my_osid_object_form._my_map['texts']['source'] = \
            self._source_metadata['default_string_values'][0]