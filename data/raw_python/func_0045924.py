def set_source(self, source_id):
        """Sets the source.

        arg:    source_id (osid.id.Id): the new publisher
        raise:  InvalidArgument - ``source_id`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``source_id`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.set_avatar_template
        if self.get_source_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_id(source_id):
            raise errors.InvalidArgument()
        self._my_map['sourceId'] = str(source_id)