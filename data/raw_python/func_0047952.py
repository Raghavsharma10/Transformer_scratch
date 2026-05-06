def set_description(self, description=None):
        """Sets a description.

        arg:    description (string): the new description
        raise:  InvalidArgument - description is invalid
        raise:  NoAccess - metadata.is_readonly() is true
        raise:  NullArgument - description is null
        compliance: mandatory - This method must be implemented.

        """
        if description is None:
            raise NullArgument()
        metadata = Metadata(**settings.METADATA['description'])
        if metadata.is_read_only():
            raise NoAccess()
        if self._is_valid_input(description, metadata, array=False):
            self._my_map['description']['text'] = description
        else:
            raise InvalidArgument