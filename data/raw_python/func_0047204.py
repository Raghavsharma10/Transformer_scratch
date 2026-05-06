def alias_proficiency(self, proficiency_id, alias_id):
        """Adds an ``Id`` to a ``Proficiency`` for the purpose of creating compatibility.

        The primary ``Id`` of the ``Proficiency`` is determined by the
        provider. The new ``Id`` performs as an alias to the primary
        ``Id``. If the alias is a pointer to another proficiency, it is
        reassigned to the given proficiency ``Id``.

        arg:    proficiency_id (osid.id.Id): the ``Id`` of a
                ``Proficiency``
        arg:    alias_id (osid.id.Id): the alias ``Id``
        raise:  AlreadyExists - ``alias_id`` is already assigned
        raise:  NotFound - ``proficiency_id`` not found
        raise:  NullArgument - ``proficiency_id`` or ``alias_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.alias_resources_template
        self._alias_id(primary_id=proficiency_id, equivalent_id=alias_id)