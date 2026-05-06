def set_display_name(self, display_name=None):
        """Sets a display name.

        A display name is required and if not set, will be set by the
        provider.

        arg:    displayName (string): the new display name
        raise:  InvalidArgument - displayName is invalid
        raise:  NoAccess - metadata.is_readonly() is true
        raise:  NullArgument - displayName is null
        compliance: mandatory - This method must be implemented.

        """
        if display_name is None:
            raise NullArgument()
        metadata = Metadata(**settings.METADATA['display_name'])
        if metadata.is_read_only():
            raise NoAccess()
        if self._is_valid_input(display_name, metadata, array=False):
            self._my_map['displayName']['text'] = display_name
        else:
            raise InvalidArgument