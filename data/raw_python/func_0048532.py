def clear_droppables(self):
        """stub"""
        if self.get_droppables_metadata().is_read_only():
            raise NoAccess()
        self.my_osid_object_form._my_map['droppables'] = \
            self._droppables_metadata['default_object_values'][0]