def set_public_domain(self, public_domain):
        """Sets the public domain flag.

        arg:    public_domain (boolean): the public domain status
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.set_group_template
        if self.get_public_domain_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_boolean(public_domain):
            raise errors.InvalidArgument()
        self._my_map['publicDomain'] = public_domain