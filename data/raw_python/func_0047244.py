def set_display_label(self, display_label):
        """Seta a display label.

        arg:    display_label (string): the new display label
        raise:  InvalidArgument - ``display_label`` is invalid
        raise:  NoAccess - ``display_label`` cannot be modified
        raise:  NullArgument - ``display_label`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.get_display_label_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_string(display_label,
                                     self.get_display_label_metadata()):
            raise errors.InvalidArgument()
        self._my_map['displayLabel']['text'] = display_label