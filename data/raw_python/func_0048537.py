def clear_targets(self):
        """stub"""
        if self.get_targets_metadata().is_read_only():
            raise NoAccess()
        self.my_osid_object_form._my_map['targets'] = \
            self._targets_metadata['default_object_values'][0]