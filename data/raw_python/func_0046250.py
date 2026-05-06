def set_public_domain(self, public_domain=None):
        """Sets the public domain flag.

        :param public_domain: the public domain status
        :type public_domain: ``boolean``
        :raise: ``NoAccess`` -- ``Metadata.isReadOnly()`` is ``true``

        *compliance: mandatory -- This method must be implemented.*

        """
        if public_domain is None:
            raise NullArgument()
        metadata = Metadata(**settings.METADATA['public_domain'])
        if metadata.is_read_only():
            raise NoAccess()
        if self._is_valid_input(public_domain, metadata, array=False):
            self._my_map['publicDomain'] = public_domain
        else:
            raise InvalidArgument()