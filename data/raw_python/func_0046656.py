def alias_authorization(self, authorization_id, alias_id):
        """Adds an ``Id`` to an ``Authorization`` for the purpose of creating compatibility.

        The primary ``Id`` of the ``Authorization`` is determined by the
        provider. The new ``Id`` performs as an alias to the primary
        ``Id``. If the alias is a pointer to another authorization. it
        is reassigned to the given authorization ``Id``.

        arg:    authorization_id (osid.id.Id): the ``Id`` of an
                ``Authorization``
        arg:    alias_id (osid.id.Id): the alias ``Id``
        raise:  AlreadyExists - ``alias_id`` is already assigned
        raise:  NotFound - ``authorization_id`` not found
        raise:  NullArgument - ``authorization_id`` or ``alias_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.alias_resources_template
        self._alias_id(primary_id=authorization_id, equivalent_id=alias_id)