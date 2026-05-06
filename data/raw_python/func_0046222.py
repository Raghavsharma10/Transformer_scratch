def clear_max_attempts(self):
        """stub"""
        if (self.get_max_attempts_metadata().is_read_only() or
                self.get_max_attempts_metadata().is_required()):
            raise NoAccess()
        self.my_osid_object_form._my_map['maxAttempts'] = \
            list(self._max_attempts_metadata['default_integer_values'])[0]