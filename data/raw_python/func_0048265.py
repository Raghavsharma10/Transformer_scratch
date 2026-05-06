def get_resource_search_session_for_bin(self, bin_id):
        """Gets a resource search session for the given bin.

        arg:    bin_id (osid.id.Id): the ``Id`` of the bin
        return: (osid.resource.ResourceSearchSession) - ``a
                ResourceSearchSession``
        raise:  NotFound - ``bin_id`` not found
        raise:  NullArgument - ``bin_id`` is ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  Unimplemented - ``supports_resource_search()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_resource_search()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_resource_search():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.ResourceSearchSession(bin_id, runtime=self._runtime)