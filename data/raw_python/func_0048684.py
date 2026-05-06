def clear_group(self):
        """Clears the group designation.

        raise:  NoAccess - ``Metadata.isRequired()`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.clear_group_template
        if (self.get_group_metadata().is_read_only() or
                self.get_group_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['group'] = self._group_default