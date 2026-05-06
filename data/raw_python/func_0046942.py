def get_composition_admin_session_for_repository(self, repository_id, proxy):
        """Gets a composiiton administrative session for the given repository.

        arg:    repository_id (osid.id.Id): the ``Id`` of the repository
        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.repository.CompositionAdminSession) - a
                ``CompositionAdminSession``
        raise:  NotFound - ``repository_id`` not found
        raise:  NullArgument - ``repository_id`` or ``proxy`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_composition_admin()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_composition_admin()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_composition_admin():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.CompositionAdminSession(repository_id, proxy, self._runtime)