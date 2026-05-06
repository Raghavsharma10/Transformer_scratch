def clear_zones(self):
        """stub"""
        if self.get_zones_metadata().is_read_only():
            raise NoAccess()
        self.my_osid_object_form._my_map['zones'] = \
            self._zones_metadata['default_object_values'][0]