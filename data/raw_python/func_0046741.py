def alias_gradebook_column(self, gradebook_column_id, alias_id):
        """Adds an ``Id`` to a ``GradebookColumn`` for the purpose of creating compatibility.

        The primary ``Id`` of the ``GradebookColumn`` is determined by
        the provider. The new ``Id`` performs as an alias to the primary
        ``Id``. If the alias is a pointer to another gradebook column,
        it is reassigned to the given gradebook column ``Id``.

        arg:    gradebook_column_id (osid.id.Id): the ``Id`` of a
                ``GradebookColumn``
        arg:    alias_id (osid.id.Id): the alias ``Id``
        raise:  AlreadyExists - ``alias_id`` is already assigned
        raise:  NotFound - ``gradebook_column_id`` not found
        raise:  NullArgument - ``gradebook_column_id`` or ``alias_id``
                is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.alias_resources_template
        self._alias_id(primary_id=gradebook_column_id, equivalent_id=alias_id)