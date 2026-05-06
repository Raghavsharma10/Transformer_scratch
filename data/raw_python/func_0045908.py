def clear_public_domain(self):
        """Removes the public domain status.

        raise:  NoAccess - ``Metadata.isRequired()`` is ``true`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.clear_group_template
        if (self.get_public_domain_metadata().is_read_only() or
                self.get_public_domain_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['publicDomain'] = self._public_domain_default