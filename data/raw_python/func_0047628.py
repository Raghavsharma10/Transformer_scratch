def clear_texts(self):
        """stub"""
        if self.get_texts_metadata().is_read_only():
            raise NoAccess()
        self.my_osid_object_form._my_map['texts'] = \
            self._texts_metadata['default_object_values'][0]