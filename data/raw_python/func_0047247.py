def clear_description(self):
        """Clears the description.

        raise:  NoAccess - ``description`` cannot be modified
        *compliance: mandatory -- This method must be implemented.*

        """
        if (self.get_domain_metadata().is_read_only() or
                self.get_domain_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['domain'] = self._domain_metadata['default_string_values'][0]