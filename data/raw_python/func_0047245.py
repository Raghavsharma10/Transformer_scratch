def clear_display_label(self):
        """Clears the display label.

        raise:  NoAccess - ``display_label`` cannot be modified
        *compliance: mandatory -- This method must be implemented.*

        """
        if (self.get_display_label_metadata().is_read_only() or
                self.get_display_label_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['displayLabel'] = self._display_label_metadata['default_string_values'][0]