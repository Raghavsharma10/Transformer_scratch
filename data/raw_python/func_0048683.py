def set_group(self, group):
        """Sets the resource as a group.

        arg:    group (boolean): ``true`` if this resource is a group,
                ``false`` otherwise
        raise:  InvalidArgument - ``group`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.set_group_template
        if self.get_group_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_boolean(group):
            raise errors.InvalidArgument()
        self._my_map['group'] = group