def set_description(self, description):
        """Sets a description.

        arg:    description (string): the new description
        raise:  InvalidArgument - ``description`` is invalid
        raise:  NoAccess - ``description`` cannot be modified
        raise:  NullArgument - ``description`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.get_description_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_string(description,
                                     self.get_description_metadata()):
            raise errors.InvalidArgument()
        self._my_map['description']['text'] = description