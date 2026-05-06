def alias_assessment(self, assessment_id, alias_id):
        """Adds an ``Id`` to an ``Assessment`` for the purpose of creating compatibility.

        The primary ``Id`` of the ``Assessment`` is determined by the
        provider. The new ``Id`` is an alias to the primary ``Id``. If
        the alias is a pointer to another assessment, it is reassigned
        to the given assessment ``Id``.

        arg:    assessment_id (osid.id.Id): the ``Id`` of an
                ``Assessment``
        arg:    alias_id (osid.id.Id): the alias ``Id``
        raise:  AlreadyExists - ``alias_id`` is in use as a primary
                ``Id``
        raise:  NotFound - ``assessment_id`` not found
        raise:  NullArgument - ``assessment_id`` or ``alias_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.alias_resources_template
        self._alias_id(primary_id=assessment_id, equivalent_id=alias_id)