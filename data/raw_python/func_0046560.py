def alias_relationship(self, relationship_id, alias_id):
        """Adds an ``Id`` to a ``Relationship`` for the purpose of creating compatibility.

        The primary ``Id`` of the ``Relationship`` is determined by the
        provider. The new ``Id`` performs as an alias to the primary
        ``Id``. If the alias is a pointer to another relationship, it is
        reassigned to the given relationship ``Id``.

        arg:    relationship_id (osid.id.Id): the ``Id`` of a
                ``Relationship``
        arg:    alias_id (osid.id.Id): the alias ``Id``
        raise:  AlreadyExists - ``alias_id`` is already assigned
        raise:  NotFound - ``relationship`` not found
        raise:  NullArgument - ``relationship_id`` or ``alias_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.alias_resources_template
        self._alias_id(primary_id=relationship_id, equivalent_id=alias_id)