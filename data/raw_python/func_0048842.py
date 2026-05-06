def alias_assessment_part(self, assessment_part_id, alias_id):
        """Adds an ``Id`` to an ``AssessmentPart`` for the purpose of creating compatibility.

        The primary ``Id`` of the ``AssessmentPart`` is determined by
        the provider. The new ``Id`` is an alias to the primary ``Id``.
        If the alias is a pointer to another assessment part, it is
        reassigned to the given assessment part ``Id``.

        arg:    assessment_part_id (osid.id.Id): the ``Id`` of an
                ``AssessmentPart``
        arg:    alias_id (osid.id.Id): the alias ``Id``
        raise:  AlreadyExists - ``alias_id`` is in use as a primary
                ``Id``
        raise:  NotFound - ``assessment_part_id`` not found
        raise:  NullArgument - ``assessment_part_id`` or ``alias_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.alias_resources_template
        self._alias_id(primary_id=assessment_part_id, equivalent_id=alias_id)