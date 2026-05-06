def set_taker(self, resource_id):
        """Sets the resource who will be taking this assessment.

        arg:    resource_id (osid.id.Id): the resource Id
        raise:  InvalidArgument - ``resource_id`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.set_avatar_template
        if self.get_taker_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_id(resource_id):
            raise errors.InvalidArgument()
        self._my_map['takerId'] = str(resource_id)