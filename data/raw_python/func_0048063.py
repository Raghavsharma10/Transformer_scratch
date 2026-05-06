def alias_assessment_offered(self, assessment_offered_id, alias_id):
        """Adds an ``Id`` to an ``AssessmentOffered`` for the purpose of creating compatibility.

        The primary ``Id`` of the ``AssessmentOffered`` is determined by
        the provider. The new ``Id`` is an alias to the primary ``Id``.
        If the alias is a pointer to another assessment offered, it is
        reassigned to the given assessment offered ``Id``.

        arg:    assessment_offered_id (osid.id.Id): the ``Id`` of an
                ``AssessmentOffered``
        arg:    alias_id (osid.id.Id): the alias ``Id``
        raise:  AlreadyExists - ``alias_id`` is in use as a primary
                ``Id``
        raise:  NotFound - ``assessment_offered_id`` not found
        raise:  NullArgument - ``assessment_offered_id`` or ``alias_id``
                is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.alias_resources_template
        self._alias_id(primary_id=assessment_offered_id, equivalent_id=alias_id)