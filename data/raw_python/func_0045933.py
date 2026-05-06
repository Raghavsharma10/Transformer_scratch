def set_published(self, published):
        """Sets the published status.

        arg:    published (boolean): the published status
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.set_group_template
        if self.get_published_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_boolean(published):
            raise errors.InvalidArgument()
        self._my_map['published'] = published