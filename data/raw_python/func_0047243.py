def clear_display_name(self):
        """Clears the display name.

        raise:  NoAccess - ``display_name`` cannot be modified
        *compliance: mandatory -- This method must be implemented.*

        """
        if (self.get_display_name_metadata().is_read_only() or
                self.get_display_name_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['displayName'] = self._display_name_metadata['default_string_values'][0]