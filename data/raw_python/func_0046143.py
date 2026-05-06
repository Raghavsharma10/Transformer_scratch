def add_display_name(self, display_name):
        """Adds a display_name.

        arg:    display_name (displayText): the new display name
        raise:  InvalidArgument - ``display_name`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``display_name`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.get_display_names_metadata().is_read_only():
            raise NoAccess()
        if not isinstance(display_name, DisplayText):
            raise InvalidArgument('display_name must be instance of DisplayText')
        self.add_or_replace_value('displayNames', display_name)