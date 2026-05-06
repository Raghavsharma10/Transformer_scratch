def alias_resource(self, resource_id, alias_id):
        """Adds an ``Id`` to a ``Resource`` for the purpose of creating compatibility.

        The primary ``Id`` of the ``Resource`` is determined by the
        provider. The new ``Id`` performs as an alias to the primary
        ``Id``. If the alias is a pointer to another resource it is
        reassigned to the given resource ``Id``.

        arg:    resource_id (osid.id.Id): the ``Id`` of a ``Resource``
        arg:    alias_id (osid.id.Id): the alias ``Id``
        raise:  AlreadyExists - ``alias_id`` is already assigned
        raise:  NotFound - ``resource_id`` not found
        raise:  NullArgument - ``alias_id`` or ``resource_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.alias_resources_template
        self._alias_id(primary_id=resource_id, equivalent_id=alias_id)