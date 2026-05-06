def alias_repository(self, repository_id=None, alias_id=None):
        """Adds an ``Id`` to a ``Repository`` for the purpose of creating compatibility.

        The primary ``Id`` of the ``Repository`` is determined by the
        provider. The new ``Id`` is an alias to the primary ``Id``. If
        the alias is a pointer to another repository, it is reassigned
        to the given repository ``Id``.

        arg:    repository_id (osid.id.Id): the ``Id`` of a
                ``Repository``
        arg:    alias_id (osid.id.Id): the alias ``Id``
        raise:  AlreadyExists - ``alias_id`` is in use as a primary
                ``Id``
        raise:  NotFound - ``repository_id`` not found
        raise:  NullArgument - ``repository_id`` or ``alias_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from awsosid template for -
        # osid.resource.BinAdminSession.alias_bin_template
        if not self._can('alias'):
            raise PermissionDenied()
        else:
            return self._provider_session.alias_repository(repository_id)