def clear_texts(self):
        """stub"""
        if self._texts_metadata['required'] or self._texts_metadata['read_only']:
            raise NoAccess()
        self.my_osid_object_form._my_map['texts'] = \
            dict(self._texts_metadata['default_object_values'][0])