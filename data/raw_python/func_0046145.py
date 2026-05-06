def add_description(self, description):
        """Adds a description.

        arg:    description (displayText): the new description
        raise:  InvalidArgument - ``description`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``description`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.get_descriptions_metadata().is_read_only():
            raise NoAccess()
        if not isinstance(description, DisplayText):
            raise InvalidArgument('description must be instance of DisplayText')
        self.add_or_replace_value('descriptions', description)