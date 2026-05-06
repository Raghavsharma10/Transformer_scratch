def alias_grade_entry(self, grade_entry_id, alias_id):
        """Adds an ``Id`` to a ``GradeEntry`` for the purpose of creating compatibility.

        The primary ``Id`` of the ``GradeEntry`` is determined by the
        provider. The new ``Id`` performs as an alias to the primary
        ``Id``. If the alias is a pointer to another grade entry, it is
        reassigned to the given grade entry ``Id``.

        arg:    grade_entry_id (osid.id.Id): the ``Id`` of a
                ``GradeEntry``
        arg:    alias_id (osid.id.Id): the alias ``Id``
        raise:  AlreadyExists - ``alias_id`` is already assigned
        raise:  NotFound - ``grade_entry_id`` not found
        raise:  NullArgument - ``grade_entry_id`` or ``alias_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.alias_resources_template
        self._alias_id(primary_id=grade_entry_id, equivalent_id=alias_id)