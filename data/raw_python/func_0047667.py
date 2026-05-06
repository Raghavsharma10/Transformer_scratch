def clear_published(self):
        """stub"""
        if (self.get_published_metadata().is_read_only() or
                self.get_published_metadata().is_required()):
            raise NoAccess()
        self.my_osid_object_form._my_map['published'] = \
            self._published_metadata['default_published_values'][0]