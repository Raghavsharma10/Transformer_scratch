def clear_unlock_previous(self):
        """stub"""
        if (self.get_unlock_previous_metadata().is_read_only() or
                self.get_unlock_previous_metadata().is_required()):
            raise NoAccess()
        self.my_osid_object_form._my_map['unlockPrevious'] = \
            str(self._unlock_previous_metadata['default_string_values'][0])