def set_display_name(self, display_name):
        """Sets a display name.

        arg:    display_name (string): the new display name
        raise:  InvalidArgument - ``display_name`` is invalid
        raise:  NoAccess - ``display_name`` cannot be modified
        raise:  NullArgument - ``display_name`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.get_display_name_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_string(display_name,
                                     self.get_display_name_metadata()):
            raise errors.InvalidArgument()
        self._my_map['displayName']['text'] = display_name