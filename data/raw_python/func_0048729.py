def add_media_description(self, media_description):
        """Adds a media_description.

        arg:    media_description (displayText): the new media_description
        raise:  InvalidArgument - ``media_description`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``media_description`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.get_media_descriptions_metadata().is_read_only():
            raise NoAccess()
        self.add_or_replace_value('mediaDescriptions', media_description)